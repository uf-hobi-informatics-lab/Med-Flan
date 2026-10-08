import logging
import warnings
from dataclasses import dataclass

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorMixin

from accelerate import PartialState
from trl import SFTTrainer
from trl.trainer.utils import pad


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForPlw(DataCollatorMixin):
    pad_token_id: int
    completion_only_loss: bool = False
    return_position_ids: bool = True
    return_tensors: str = "pt"

    def torch_call(self, examples):
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [
            torch.tensor(example["attention_mask"]) for example in examples
        ]
        labels = [torch.tensor(example["input_ids"]) for example in examples]
        prompt_mask = [torch.tensor(example["prompt_mask"]) for example in examples]
        completion_mask = [
            torch.tensor(example["completion_mask"]) for example in examples
        ]

        # Pad
        output = {}
        output["input_ids"] = pad(
            input_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        output["attention_mask"] = pad(
            attention_mask, padding_value=0, padding_side="right"
        )
        output["labels"] = pad(labels, padding_value=-100, padding_side="right")
        output["prompt_mask"] = pad(prompt_mask, padding_value=0, padding_side="right")
        output["completion_mask"] = pad(
            completion_mask, padding_value=0, padding_side="right"
        )

        return output


def PLW_sample_chat_template():
    template = "{% if messages[0]['role'] == 'system' %}{% set system_message =  bos_token + '### System Instruction: ' + messages[0]['content'] | trim + '' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ system_message }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Context: ' + message['content'] | trim + '\n' }}{% elif message['role'] == 'assistant' %}{{ '### Result: ' + message['content'] | trim + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Result: ' }}{% endif %}"

    return template


def extract_assistant_text(
    tokenizer, messages, enable_thinking=None, use_generation_prompt=False
):
    # enable_thinking (Qwen3) only affects generation-prompt rendering; full always uses
    # add_generation_prompt=False (the complete conversation). use_generation_prompt
    # controls the prefix boundary: False (default) -> prefix is everything before the
    # assistant turn, so completion = the full assistant turn (header + think block +
    # answer + eos); True -> prefix uses add_generation_prompt=True, so the assistant
    # header + (Qwen3 no-think) empty think block are pre-filled in the prefix and
    # completion = answer + eos only -- exactly matching the inference generation target.
    # The full token sequence is identical either way; only the prompt/completion
    # (loss-weight) boundary moves.
    template_kwargs = (
        {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **template_kwargs
    )

    # Everything before assistant (system + user), optionally + the generation prompt
    prefix = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=use_generation_prompt,
        **template_kwargs,
    )

    # Strip prefix to get assistant-only content
    return full[len(prefix) :]


def PLW_apply_chat_template(
    example,
    tokenizer=None,
    use_sample_template=False,
    enable_thinking=None,
    use_generation_prompt=False,
):
    messages = example["messages"]
    prompts = list(filter(lambda x: x["role"] != "assistant", messages))
    labels = list(filter(lambda x: x["role"] == "assistant", messages))
    # add empty user msg
    labels = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
    ] + labels

    if use_sample_template:
        tokenizer.chat_template = PLW_sample_chat_template()

    template_kwargs = (
        {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
    )
    example["prompt"] = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=use_generation_prompt,
        **template_kwargs,
    )

    # completion_text_after_apply_template = tokenizer.apply_chat_template(
    #     labels, tokenize=False, add_generation_prompt=False
    # )
    completion_text_after_apply_template = extract_assistant_text(
        tokenizer,
        labels,
        enable_thinking=enable_thinking,
        use_generation_prompt=use_generation_prompt,
    )
    example["completion"] = completion_text_after_apply_template
    # deal with qwen 2.5 models where it chat template insert system message in all cases
    # if "Qwen2.5" in tokenizer.name_or_path:
    #     index = completion_text_after_apply_template.rindex("<|im_start|>")
    #     # remove system prompt
    #     example["completion"] = completion_text_after_apply_template[index:]
    # else:
    #     example["completion"] = completion_text_after_apply_template
    return example


class PLWTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        prompt_loss_weight=0.1,
        **kwargs,
    ):
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.plw = prompt_loss_weight
        # use plw data collator or data collator from transformers
        # currently use transformer DataCollatorForLanguageModeling as data collator
        # need to test DataCollatorForPlw and use this one instead to support batch > 1
        tokenizer = kwargs.get("processing_class", None)
        if tokenizer is None:
            tokenizer = kwargs.get("tokenizer", None)

        data_collator = DataCollatorForPlw(pad_token_id=tokenizer.pad_token_id)
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # need to add prompt_mask and completion_mask to dataset generation
        super().__init__(*args, data_collator=data_collator, **kwargs)

    def _set_signature_columns_if_needed(self):
        # TRL's SFTTrainer hardcodes self._signature_columns to
        # ["input_ids", "labels", "seq_lengths"], which doesn't know about the
        # extra columns PLWTrainer's collator/compute_loss need. With
        # remove_unused_columns: true (the default in our recipes), the base
        # Trainer strips any dataset column not in that list before collation --
        # silently dropping attention_mask, prompt_mask, and completion_mask and
        # crashing with KeyError: 'attention_mask' in DataCollatorForPlw.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "labels",
                "prompt_mask",
                "completion_mask",
            ]

    def get_ref_model_output(self, model, inputs):
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        return outputs

    def _activation_checkpointing_active(self):
        # use_cache=True is incompatible with any form of activation/gradient
        # checkpointing: the recomputed forward produces attention K/V tensors with
        # different shapes than the saved forward (e.g. saved [1,125,8,128] vs recomputed
        # [1,375,8,128]) -> torch.utils.checkpoint.CheckpointError in backward. Detect both
        # checkpointing sources: (1) HF's gradient_checkpointing flag, and (2) FSDP's
        # *separate* activation_checkpointing flag -- the qwen/ouro recipes set
        # gradient_checkpointing: false and use FSDP activation_checkpointing: true instead,
        # so a plain `not gradient_checkpointing` guard missed it and let use_cache=True
        # through under active FSDP checkpointing (the bug). TRL's std SFTTrainer sidesteps
        # this by unconditionally forcing use_cache=False (sft_trainer.py:1773); PLW honors
        # the yaml use_cache field only when no checkpointing is active.
        if self.args.gradient_checkpointing:
            return True
        plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        return bool(
            plugin is not None and getattr(plugin, "activation_checkpointing", False)
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if self.model.training else "eval"

        # Pop labels/prompt_mask/completion_mask *before* calling the model: TRL's
        # default chunked_nll forward patch detects `labels` in the call and skips
        # materializing logits entirely (it computes the loss internally instead),
        # leaving outputs.logits as None. PLWTrainer needs real logits for its own
        # prompt/completion-weighted loss, so labels must never reach model.forward.
        labels = inputs.pop("labels")
        prompt_mask = inputs.pop("prompt_mask")
        completion_mask = inputs.pop("completion_mask")

        # use_cache is a yaml-settable TrainingArguments field (default overridden to False
        # in our SFTConfig so PLW runs cache-free by default, matching TRL's std SFTTrainer
        # which forces use_cache=False at forward time; Ouro's recurrence makes the cache a
        # 4x multiplier and must stay off). Apply at forward time -- mirrors TRL's
        # SFTTrainer.compute_loss forcing use_cache=False (sft_trainer.py:1773) -- so it
        # overrides the model's loaded config regardless of how the model accepts the kwarg.
        # Activation/gradient checkpointing requires use_cache=False (a hard torch invariant:
        # use_cache=True under checkpointing raises CheckpointError in backward from
        # mismatched recomputed K/V shapes), so force it off under either checkpointing source
        # (HF gradient_checkpointing OR FSDP activation_checkpointing -- see
        # _activation_checkpointing_active) even if a recipe left use_cache: true.
        inputs["use_cache"] = (
            self.args.use_cache and not self._activation_checkpointing_active()
        )

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs, return_dict=True)

        logits = outputs.get("logits")
        weights = self.plw * prompt_mask + completion_mask

        # Slice once into a contiguous `flat` and reuse it for both the weighted CE and the
        # argmax -- avoids the redundant internal copy argmax would make on a non-contiguous
        # slice if CE and argmax sliced `logits` separately. No chunking: chunking the CE call
        # alone does NOT reduce peak (CrossEntropyLoss saves its input for backward, so
        # retention stays ~B*S*V); the real peak-memory fix is the Liger/checkpoint path
        # (Phase D), not here. Shift by slicing the seq dim.
        V = logits.size(-1)
        flat = logits[:, :-1, :].reshape(-1, V)
        shift_labels = labels[:, 1:]
        shift_weights = weights[:, 1:]
        shift_labels = shift_labels.to(flat.device)
        shift_weights = shift_weights.to(flat.device, flat.dtype)

        # Compute per-token losses (order (b, s) -- aligns with the masks' reshape(-1))
        token_losses = self.loss_fct(flat, shift_labels.reshape(-1))

        # Gradient-accumulation normalization. HF Trainer skips its loss / gas
        # division when `model_accepts_loss_kwargs` is True AND num_items_in_batch
        # is provided (transformers/trainer.py:1952 -- our installed 5.15.1; the
        # shipped Trainer docstring explicitly warns to set model_accepts_loss_kwargs
        # False if you don't normalize by num_items_in_batch). Qwen3/Ouro accept
        # loss kwargs, so that branch fires: the Trainer does NOT divide our
        # returned loss by gradient_accumulation_steps. PLW's per-microbatch
        # weighted mean (÷ shift_weights.sum()) is only exact at gas==1; at gas>1
        # the micro-batch means would accumulate to gas× the global mean, i.e. an
        # effective learning rate gas× too high (ouro-PLW runs gas=4). Divide by the
        # global non-ignored-token count (num_items_in_batch) in that regime instead
        # -- the same normalization std SFTTrainer uses -- so the summed micro-batch
        # losses equal the global weighted sum over the global token count. Gate on
        # gas>1 so gas==1 (all qwen PLW recipes) keeps its tuned weight-sum
        # normalization byte-for-byte; gate on model_accepts_loss_kwargs so an older
        # transformers that lacks it (or a model that doesn't accept loss kwargs)
        # falls back to the per-microbatch mean, which the Trainer then divides by
        # gas into the bounded "mean of means".
        accumulating = self.args.gradient_accumulation_steps > 1
        new_regime = bool(getattr(self, "model_accepts_loss_kwargs", False))
        if accumulating and new_regime and num_items_in_batch is not None:
            denom = num_items_in_batch
        else:
            denom = shift_weights.sum()
        loss = token_losses.float() @ shift_weights.reshape(-1).float() / (denom + 1e-9)

        # Metric denominators must match the shifted numerator masks (prompt_mask /
        # completion_mask[..., 1:]) -- the old code divided by the *unshifted*
        # mask sum, which counts position 0. Position 0 is always a prompt token
        # (the sequence starts with the prompt), so prompt_loss's denominator was
        # larger by ~B and the logged value was a slight underestimate;
        # completion_loss was unaffected (completion_mask[:, 0] is always 0) but
        # is shifted here too for self-evident parity.
        shift_prompt_mask = prompt_mask[..., 1:]
        prompt_loss = (
            token_losses.reshape(-1).float()
            @ shift_prompt_mask.reshape(-1).float()
            / (shift_prompt_mask.sum() + 1e-9)
        )
        self._metrics[mode]["prompt_loss"].append(prompt_loss.item())

        shift_completion_mask = completion_mask[..., 1:]
        completion_loss = (
            token_losses.reshape(-1).float()
            @ shift_completion_mask.reshape(-1).float()
            / (shift_completion_mask.sum() + 1e-9)
        )
        self._metrics[mode]["completion_loss"].append(completion_loss.item())
        self._metrics[mode]["full_loss"].append(loss.item())

        # Accuracy: predictions already in hand via flat.argmax (no second full-logits
        # argmax). Guard the cross-rank gather behind a step interval (default every 50;
        # 0 or 1 => every step; eval always). `interval <= 1` avoids ZeroDivisionError on
        # `step % interval` when the knob is 0.
        interval = self.args.plw_accuracy_log_steps
        if not self.args.use_liger_kernel and (
            interval <= 1 or self.state.global_step % interval == 0 or mode == "eval"
        ):
            predictions = flat.argmax(dim=-1).view(logits.size(0), -1)
            mask = shift_weights != 0
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (
                (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            )
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
        # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
        column_names = (
            dataset.column_names
            if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
            else None
        )
        if column_names and "input_ids" in column_names:
            if formatting_func is not None:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
                )

            return dataset

        # check if torch dataset / dataloader and do nothing
        # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
        if isinstance(
            dataset,
            (torch.utils.data.IterableDataset, torch.utils.data.Dataset),
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        self._dataset_sanity_checked = False

        # Resolve the length cap from args.max_length (TRL's current field name -- older
        # versions used max_seq_length; try both defensively so a version mismatch falls
        # back loudly rather than silently reproducing the old bug) rather than
        # tokenizer.model_max_length, which for e.g. Qwen3 is 131072 vs the recipe's
        # 32768 -- letting PLW process 4x-longer sequences than std SFTTrainer.
        max_len = (
            getattr(args, "max_length", None)
            or getattr(args, "max_seq_length", None)
            or processing_class.model_max_length
        )
        strategy = getattr(args, "plw_overlength_strategy", "drop")
        logger.info(f"PLW length cap: {max_len}, over-length strategy: {strategy}")

        def tokenize(element):
            prompts = element["prompt"]
            labels = element["completion"]

            input_ids = []
            attention_mask = []
            prompt_masks = []
            completion_masks = []

            for prmp, lab in zip(prompts, labels):
                p = processing_class(
                    prmp,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_overflowing_tokens=False,
                    return_length=True,
                )

                l = processing_class(
                    lab,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_overflowing_tokens=False,
                    return_length=True,
                )

                p_len = p["length"][0]
                l_len = l["length"][0]
                total_len = p_len + l_len

                if total_len > max_len:
                    if strategy == "error":
                        raise ValueError(
                            f"over-length example ({total_len} > {max_len}); set "
                            "plw_overlength_strategy to 'drop' or 'truncate_right' to handle it"
                        )
                    if strategy == "drop":
                        logger.warning(
                            f"dropping over-length example: {total_len} > {max_len}"
                        )
                        continue
                    if strategy == "truncate_right":
                        # Keep the first max_len tokens (drop the tail) -- parity with std
                        # SFTTrainer's keep_start truncation; preserves the system prompt +
                        # task instruction (unlike the old left-truncation, which dropped
                        # them). May lose the completion tail. Guard the empty-completion
                        # hazard: if max_len lands inside the prompt, the completion is
                        # gone entirely and a zero completion_mask would divide by zero in
                        # the completion_loss metric -- drop instead.
                        seq = (p["input_ids"] + l["input_ids"])[:max_len]
                        kept_p = min(p_len, max_len)
                        kept_l = len(seq) - kept_p
                        if kept_l == 0:
                            logger.warning(
                                f"truncate_right left no completion (p_len={p_len} >= "
                                f"{max_len}); dropping"
                            )
                            continue
                        input_ids.append(seq)
                        attention_mask.append([1] * len(seq))
                        prompt_masks.append([1] * kept_p + [0] * kept_l)
                        completion_masks.append([0] * kept_p + [1] * kept_l)
                        continue
                    raise ValueError(f"unknown plw_overlength_strategy: {strategy!r}")

                new_input_ids = p["input_ids"] + l["input_ids"]
                new_attn_mask = p["attention_mask"] + l["attention_mask"]
                prompt_mask = p["attention_mask"] + [0] * l_len
                completion_mask = [0] * p_len + l["attention_mask"]

                input_ids.append(new_input_ids)
                attention_mask.append(new_attn_mask)
                prompt_masks.append(prompt_mask)
                completion_masks.append(completion_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_mask": prompt_masks,
                "completion_mask": completion_masks,
            }

        before = len(dataset)
        with PartialState().main_process_first():
            dataset = dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=args.dataset_num_proc,
            )
        # NOTE: this count and the Step-0 per-task audit's over-length count need not match
        # exactly -- the audit tokenizes the full rendered template in one pass, while
        # tokenize() above tokenizes prompt and completion separately and concatenates,
        # which can differ by a token or two at the boundary. Fine for deciding a strategy;
        # don't chase the discrepancy.
        logger.info(
            f"PLW over-length handling removed/trimmed {before - len(dataset)} of "
            f"{before} examples (strategy={strategy}, cap={max_len})"
        )

        return dataset


# TODO
class PLWLossTrackingTrainer(PLWTrainer):
    def __init__(
        self, *args, prompt_loss_weight=0.1, custom_train_dataset=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(reduction="none")
        self.plw = prompt_loss_weight
        self.custom_train_dataset = custom_train_dataset
        self.current_losses = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Retrieve the original indices
        data_indices = inputs.pop("data_idx")

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Reshape loss back to per-sample loss
        # Need to handle padding carefully. A simpler approach for now is to
        # average over non-padding tokens. For true per-sample loss, you'd need
        # to mask out padding.

        # Assuming padding token is 0 or -100 for labels, filter it out
        # This is a simplification; for proper per-sequence loss,
        # you'd need to correctly sum loss for each sequence.
        # For a basic example, we'll just average the loss of tokens in each sequence.
        # This will be more accurate if all sequences in a batch are roughly the same length
        # or if you use `pack_sequences` with TRL.

        # If your data_collator pads to max_length in batch, the loss for padding tokens
        # should be ignored. The CrossEntropyLoss with ignore_index=-100 does this.
        # We need to average the loss for each sample in the batch.

        # Reshape to (batch_size, sequence_length-1)
        per_token_loss = loss.view(logits.shape[0], -1)

        # Calculate loss per sample, ignoring -100 (padding)
        # Assuming -100 is the ignore_index for labels
        valid_tokens_mask = shift_labels != -100

        # Sum valid token losses for each sample, then divide by count of valid tokens
        # Add a small epsilon to avoid division by zero
        per_sample_loss = (per_token_loss * valid_tokens_mask).sum(dim=1) / (
            valid_tokens_mask.sum(dim=1) + 1e-6
        )

        self.current_losses = {
            idx.item(): l.item() for idx, l in zip(data_indices, per_sample_loss)
        }

        # The Trainer expects a single scalar loss for backward pass
        overall_loss = per_sample_loss.mean()

        return (overall_loss, outputs) if return_outputs else overall_loss

    def get_train_dataloader(self):
        # Override this method to provide our custom DataLoader with a WeightedRandomSampler
        if self.custom_train_dataset is None:
            raise ValueError(
                "custom_train_dataset must be provided to LossTrackingTrainer"
            )

        # Create a WeightedRandomSampler based on the current weights
        sampler = WeightedRandomSampler(
            weights=self.custom_train_dataset.weights,
            num_samples=len(
                self.custom_train_dataset
            ),  # Or args.train_batch_size * num_steps_per_epoch if you want fixed number of samples
            replacement=True,  # Important for weighted sampling
        )

        # Use the default data collator if not provided
        data_collator = (
            self.data_collator if self.data_collator is not None else self.tokenizer.pad
        )

        return DataLoader(
            self.custom_train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def prepare_compute_metrics(llm_dataset, tokenizer):
    # get token ids for multiple choice options
    ABCD_token_ids = np.array(
        [
            tokenizer.convert_tokens_to_ids(x)
            for x in ["A", "B", "C", "D", "▁A", "▁B", "▁C", "▁D"]
        ]
    )

    # extract prompt/completion masks
    prompt_mask = np.array([x["prompt_mask"] for x in llm_dataset["test"]])
    completion_mask = np.array([x["completion_mask"] for x in llm_dataset["test"]])

    # uses numpy arrays (on CPU)
    def compute_metrics(data):

        # data.predictions contains the tuple (token_preds, token_losses)
        # from the preprocess_logits_for_metrics function (below)
        token_preds, token_losses = data.predictions

        # shift labels and masks
        labels = data.label_ids[..., 1:]
        shift_prompt_mask = prompt_mask[..., 1:]
        shift_comp_mask = completion_mask[..., 1:]

        # average both losses (prompt and completion) over their respective tokens
        prompt_loss = (
            token_losses.reshape(-1)
            @ shift_prompt_mask.reshape(-1)
            / shift_prompt_mask.sum()
        )
        completion_loss = (
            token_losses.reshape(-1)
            @ shift_comp_mask.reshape(-1)
            / shift_comp_mask.sum()
        )

        # compute response token accuracy
        nz = np.nonzero(shift_comp_mask)
        idx = np.where(np.isin(labels[nz], ABCD_token_ids))
        accuracy = np.mean(token_preds[nz][idx] == labels[nz][idx])

        # return metrics
        return {
            "comp_loss": completion_loss,
            "prompt_loss": prompt_loss,
            "acc": accuracy,
        }

    return compute_metrics


# uses PyTorch tensors (on GPU)
def preprocess_logits_for_metrics(logits, labels):
    # get predictions
    token_preds = logits.argmax(-1)[..., :-1]

    # compute per-token losses
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    # pass predictions and losses to compute_metrics function (above)
    predictions = (token_preds, token_losses)
    return predictions
