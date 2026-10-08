#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys
import types
from pathlib import Path


p = Path(__file__).parent.parent / "src"
sys.path.append(p.as_posix())

from importlib.metadata import version as get_pkg_version

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    PLW_apply_chat_template,
    PLWTrainer,
    SFTConfig,
    WeightedJSON_apply_chat_template,
    WeightedJSONTrainer,
    apply_chat_template,
    fix_lr_scheduler_kwargs_float,
    get_checkpoint,
    get_datasets,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    report_fallback_rates,
    sync_special_token_ids,
    tokenizer_and_embedding_resize,
)
from alignment.context_extension import get_context_extension_kwargs
from trl import SFTTrainer


logger = logging.getLogger(__name__)


def _bind_train_last_step_only_forward(model):
    """Patch a looped/Universal-Transformer CausalLM (e.g. Ouro) so that `labels`-driven
    loss is computed from only the final UT loop step's hidden state, instead of the
    model's native multi-step weighted logits combination. This matches inference behavior
    (early_exit_threshold never exits early in practice) and avoids materializing one
    full [batch, seq, vocab] logits tensor per loop step.

    Assumes `self.model(...)` returns `(outputs, hidden_states_per_step, gate_weights_per_step)`
    with `outputs.last_hidden_state` and `outputs.past_key_values`, mirroring ByteDance/Ouro's
    published modeling code. If the installed modeling_ouro.py differs, this raises immediately
    with a pointer to what to check rather than silently guessing.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        try:
            base_outputs, _hidden_states_per_step, _gate_weights_per_step = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            last_hidden_state = base_outputs.last_hidden_state
        except (TypeError, ValueError, AttributeError) as e:
            raise RuntimeError(
                "train_last_step_only patch could not unpack the underlying model's output "
                "(expected `outputs, hidden_states_per_step, gate_weights_per_step = "
                "self.model(...)` with `outputs.last_hidden_state`). The installed "
                "modeling_ouro.py return signature differs from what this patch expects -- "
                "inspect it (grep for `compute_expected_logits`/`early_exit`) and update "
                "`_bind_train_last_step_only_forward` in scripts/run_sft.py accordingly."
            ) from e

        idx = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(last_hidden_state[:, idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_outputs.past_key_values,
        )

    model.forward = types.MethodType(patched_forward, model)


def _bind_huginn_forward_patch(model, fixed_num_steps=None):
    """Patch a Huginn-style depth-recurrent CausalLM (tomg-group-umd/huginn-0125) so it
    can be trained with a standard HF Trainer/TRL SFTTrainer.

    Three upstream quirks this works around (see raven_modeling_minimal.py):
    1. `forward(..., labels=...)` computes `F.cross_entropy(logits, labels)` with NO
       internal shift -- unlike every standard HF CausalLM. Left alone, this trains an
       identity map (predict the current token) instead of next-token prediction. Fix:
       call the original forward with `labels=None` to get bare logits, then shift and
       compute the loss ourselves.
    2. Its attention path can't safely take a 2-D padding `attention_mask` for batch > 1,
       and `position_ids` handling breaks for batch > 1 too. Fix: never forward either --
       safe as long as padding is right-side (real tokens only ever attend to earlier,
       real tokens; padded positions carry label=-100 so they don't affect the loss).
    3. `output_details` defaults to `return_latents=True`, cloning a `[B, S, 5280]` tensor
       every forward for no reason during training. Fix: turn it off.

    `num_steps` is deliberately left as whatever was passed in (None from the Trainer),
    so the model's own randomized depth sampler stays active during training and eval,
    matching how Huginn was pretrained -- unless `fixed_num_steps` is given, in which case
    every forward (train and eval) is forced to that exact depth instead, bypassing the
    sampler entirely.
    """
    original_forward = model.forward

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        use_cache=None,
        past_key_values=None,
        cache_position=None,
        num_steps=None,
        **kwargs,
    ):
        effective_num_steps = (
            fixed_num_steps if fixed_num_steps is not None else num_steps
        )
        try:
            outputs = original_forward(
                input_ids=input_ids,
                labels=None,
                use_cache=False,
                num_steps=effective_num_steps,
                past_key_values=past_key_values,
                cache_position=cache_position,
                output_details={
                    "return_logits": True,
                    "return_latents": False,
                    "return_head": False,
                    "return_stats": False,
                },
                **kwargs,
            )
        except TypeError as e:
            raise RuntimeError(
                "huginn_forward_patch could not call the underlying model's forward with "
                "the expected signature (`output_details`/`num_steps` kwargs). The "
                "installed raven_modeling_minimal.py differs from what this patch expects "
                "-- inspect it and update `_bind_huginn_forward_patch` in scripts/run_sft.py "
                "accordingly."
            ) from e

        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            outputs.loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return outputs

    model.forward = types.MethodType(patched_forward, model)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()
    fix_lr_scheduler_kwargs_float(training_args)

    # `max_seq_length` is this repo's legacy field name (older TRL versions truncated on
    # it); current TRL SFTConfig truncates on `max_length` instead and never reads
    # `max_seq_length`. The installed TRL's `max_length` defaults to 1024 (not None), so
    # a `training_args.max_length is None` guard never fires -- every recipe here only
    # sets `max_seq_length`, so silently truncating at TRL's built-in 1024 instead of the
    # intended value. No recipe in this repo sets `max_length` directly, so unconditionally
    # preferring `max_seq_length` when set is safe.
    if training_args.max_seq_length is not None:
        training_args.max_length = training_args.max_seq_length

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=[
            "messages",
            "chosen",
            "rejected",
            "prompt",
            "completion",
            "label",
            "task",
        ],
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    # Truncate from left to ensure we don't lose labels in final turn
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(
        model_args, data_args, training_args, auto_set_chat_template=True
    )
    logger.info(f"tokenizer: {tokenizer}")
    if tokenizer.pad_token == tokenizer.eos_token:
        logger.warning(
            "The tokenizer's pad_token is the same as the eos_token. This may cause issues during training."
        )

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    quantization_config = (
        get_quantization_config(model_args) if model_args.use_peft else None
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if model_args.huginn_forward_patch:
        # RavenForCausalLM's custom __init__ does not accept `use_cache` as a constructor
        # kwarg (standard HF models absorb it into GenerationConfig before it reaches
        # cls(config, **model_kwargs); this custom modeling code doesn't) -- passing it
        # through from_pretrained fails with `TypeError: ... got an unexpected keyword
        # argument 'use_cache'`. The forward patch already forces use_cache=False on every
        # call, so dropping it here loses nothing.
        model_kwargs.pop("use_cache", None)

    # Context-window extension: no-op if the checkpoint's own trained max length
    # already covers training_args.max_length (e.g. Ouro-2.6B's native 65536), else
    # extends it (RoPE scaling for standard configs, block_size/rope_base for
    # Huginn/Raven-style configs) -- see alignment/context_extension.py.
    context_extension_kwargs = get_context_extension_kwargs(
        model_args.model_name_or_path,
        training_args.max_length,
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
    )
    model_kwargs.update(context_extension_kwargs)
    if "block_size" in context_extension_kwargs:
        # Huginn/Raven's freqs_cis is a *persistent* buffer, saved into the checkpoint's
        # safetensors at its original block_size. __init__ builds a fresh, correctly-sized
        # buffer from the overridden block_size/rope_base above, but from_pretrained then
        # tries to copy the checkpoint's old-shaped freqs_cis over it and crashes with a
        # state_dict size mismatch. freqs_cis is deterministic (not learned), so skipping
        # the mismatched saved tensor and keeping the freshly-computed one is correct.
        model_kwargs["ignore_mismatched_sizes"] = True

    model = model_args.model_name_or_path

    needs_eager_model = (
        model_args.train_last_step_only
        or model_args.freeze_param_patterns
        or model_args.huginn_forward_patch
        or data_args.additional_special_tokens
        or data_args.additional_non_special_tokens
    )

    if needs_eager_model:
        # Eager path: needed whenever we must touch the model object before handing it to
        # the Trainer. Used e.g. for looped/Universal-Transformer models like Ouro, or to
        # add/resize new special tokens (e.g. a dedicated pad token).
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )

        if (
            data_args.additional_special_tokens
            or data_args.additional_non_special_tokens
        ):
            tokenizer_and_embedding_resize(data_args, tokenizer, model)

        if model_args.train_last_step_only:
            _bind_train_last_step_only_forward(model)

        if model_args.huginn_forward_patch:
            if model_args.huginn_fixed_num_steps is not None and (
                model_args.huginn_mean_recurrence is not None
                or model_args.huginn_mean_backprop_depth is not None
            ):
                logger.warning(
                    "huginn_fixed_num_steps is set -- the randomized depth sampler never "
                    "runs, so huginn_mean_recurrence/huginn_mean_backprop_depth have no "
                    "effect and will be ignored."
                )
            if model_args.huginn_mean_recurrence is not None:
                model.config.mean_recurrence = model_args.huginn_mean_recurrence
            if model_args.huginn_mean_backprop_depth is not None:
                model.config.mean_backprop_depth = model_args.huginn_mean_backprop_depth
            _bind_huginn_forward_patch(
                model, fixed_num_steps=model_args.huginn_fixed_num_steps
            )

        if model_args.freeze_param_patterns:
            n_frozen = 0
            for name, param in model.named_parameters():
                if any(pat in name for pat in model_args.freeze_param_patterns):
                    param.requires_grad_(False)
                    n_frozen += 1
            logger.info(
                f"Froze {n_frozen} parameters matching {model_args.freeze_param_patterns}"
            )

        if (
            model_args.huginn_forward_patch or model_args.train_last_step_only
        ) and training_args.loss_type == "chunked_nll":
            # trl's default loss_type ("chunked_nll" unless use_liger_kernel=True)
            # patches model.forward again inside SFTTrainer.__init__ to call the
            # backbone directly and read outputs.last_hidden_state -- this silently
            # supersedes the custom forward just bound above for every step that has
            # labels, then crashes/misbehaves against a non-standard backbone output
            # (e.g. Huginn's CausalLMOutputRecurrentLatents has no last_hidden_state).
            logger.warning(
                "loss_type='chunked_nll' would override the huginn_forward_patch/"
                "train_last_step_only forward patch bound above and is incompatible "
                "with this model's non-standard backbone output. Forcing "
                "loss_type='nll' instead."
            )
            training_args.loss_type = "nll"
    else:
        # Original path: model stays a path string; TRL's SFTTrainer/PLWTrainer
        # instantiates it lazily via training_args.model_init_kwargs.
        training_args.model_init_kwargs = model_kwargs

    #####################
    # Apply chat template
    #####################
    logger.info("*** apply chat template ***")

    assert not (training_args.use_plw and training_args.use_weighted_json_loss), (
        "use_plw and use_weighted_json_loss are mutually exclusive -- both restructure "
        "the prompt/completion loss masking."
    )

    if training_args.use_plw:
        processed_datasets = raw_datasets.map(
            PLW_apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "use_sample_template": training_args.use_plw_sample_template,
                "enable_thinking": training_args.enable_thinking,
                "use_generation_prompt": training_args.plw_match_inference_boundary,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )
    elif training_args.use_weighted_json_loss:
        processed_datasets = raw_datasets.map(
            WeightedJSON_apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "enable_thinking": training_args.enable_thinking,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[c for c in column_names if c != "task"],
            desc="Applying chat template",
        )
        report_fallback_rates(processed_datasets, threshold=0.01)
        processed_datasets = processed_datasets.remove_columns(["_wjl_fallback"])
    else:
        processed_datasets = raw_datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
                "role_map": data_args.role_map,
                "enable_thinking": training_args.enable_thinking,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Applying chat template",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]

    # this is hard coded - move to config.yaml
    training_args.dataset_text_field = "text"

    ########################
    # Initialize the Trainer
    ########################
    peft_config = get_peft_config(model_args)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if get_pkg_version("transformers") >= "5.0.0" or get_pkg_version("trl") >= "0.16.0":
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if get_pkg_version("trl") < "0.13.0":
        trainer_kwargs["dataset_kwargs"] = training_args.dataset_kwargs

    if training_args.use_plw:
        trainer_kwargs["prompt_loss_weight"] = training_args.prompt_loss_weight

    if training_args.use_plw:
        trainer = PLWTrainer(**trainer_kwargs)
    elif training_args.use_weighted_json_loss:
        trainer = WeightedJSONTrainer(**trainer_kwargs)
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    if data_args.special_tokens:
        sync_special_token_ids(trainer.model, tokenizer)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logger.info(f"Checkpoint detected, resuming training at {checkpoint}.")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        # "finetuned_from": model_args.model_name_or_path, # modify for sfttrainer latest update
        "model_name": model_args.model_name_or_path,
        "dataset_name": "\n".join(list(data_args.dataset_mixer.keys())),
        "tags": ["alignment-handbook"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # torch.cuda.memory._dump_snapshot(Path(training_args.output_dir) / "GPU_RAM_PROFILE.pickle")
    # prof.close()
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history()
    main()
