"""Per-token weighted cross-entropy for JSON-formatted SFT targets (med multitask mix).

Plain per-token CE over the rendered assistant turn treats every character the same:
`{`, `"`, required keys, and separators get exactly as much gradient weight as the
tokens that actually carry the answer (a value, an optional key's presence, a
list-continuation/stop choice). That dilutes signal and skews per-task loss
comparisons across a multitask mix of varying task difficulty.

This module classifies every character of a task's canonical JSON target as either
`SCAFFOLD` (mandatory punctuation/required-key boilerplate -- no real choice was
made) or `DECISION` (a value, an optional key's presence, an array continue/stop, the
final stop token) using the *declared schema* in `med_target_schema.py`, not the
instance -- see `WeightedJSONWriter` docstring for why an instance-driven rule is
wrong. Classes, not weights, are stored per example; `token_weights_from_offsets`
converts class -> float weight at `_prepare_dataset` time using
`training_args.weighted_json_scaffold_weight`, so sweeping that hyperparameter never
requires re-deriving spans.

Run the self-check with `python -m alignment.weighted_json_loss`: every task in
`med_target_schema.SPECS` round-trips through the writer and re-parses back to its
canonicalized input.
"""

import json
import logging
from dataclasses import dataclass

import datasets
import torch
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import DataCollatorMixin

from accelerate import PartialState
from trl import SFTTrainer
from trl.trainer.utils import pad

from .med_target_schema import (
    SPECS,
    Arr,
    CoupledObj,
    DynObj,
    Enum,
    Num,
    Obj,
    Str,
    canonicalize,
)
from .plw_trainer import extract_assistant_text


logger = logging.getLogger(__name__)

SCAFFOLD = 0
DECISION = 1


class SchemaMismatch(Exception):
    """Raised when a target's shape doesn't match its task's declared spec."""


# --- writer -----------------------------------------------------------------------


def _write_generic_value(value, pieces):
    """Fallback classifier for subtrees with no schema node (unknown task, or an
    unexpected/extra key not in the spec): every character is a decision except the
    mandatory JSON punctuation -- i.e. behaves like today's ungated CE."""
    text = json.dumps(value, ensure_ascii=False)
    punct = set('{}[]":, ')
    run_text, run_cls = "", None
    for ch in text:
        cls = SCAFFOLD if ch in punct else DECISION
        if cls == run_cls:
            run_text += ch
        else:
            if run_cls is not None:
                pieces.append((run_text, run_cls))
            run_text, run_cls = ch, cls
    if run_cls is not None:
        pieces.append((run_text, run_cls))


def _write_string_value(value, pieces, nullable=False):
    if value is None:
        pieces.append(("null", DECISION))
        return
    text = json.dumps(value, ensure_ascii=False)
    pieces.append(('"', SCAFFOLD))
    pieces.append((text[1:-1], DECISION))
    pieces.append(('"', SCAFFOLD))


def _write_num_value(value, pieces, nullable=False):
    if value is None:
        pieces.append(("null", DECISION))
        return
    pieces.append((json.dumps(value), DECISION))


def _write_arr(item_spec, value, pieces):
    if not isinstance(value, list):
        raise SchemaMismatch(f"expected array, got {type(value).__name__}")
    pieces.append(("[", SCAFFOLD))
    n = len(value)
    if n == 0:
        # Stopping immediately (an empty array) is itself a choice.
        pieces.append(("]", DECISION))
        return
    for i, v in enumerate(value):
        _write(item_spec, v, pieces)
        # Continue-vs-stop is always a real choice: another item could always follow,
        # and stopping (closing the array) is a choice made at every position.
        pieces.append((", " if i < n - 1 else "]", DECISION))


def _trailing_has_choice(items, idx, key_order, required):
    """Whether the separator/closing-brace following `items[idx]` is a decision:
    true iff, in canonical order, some key that could legally appear after this one
    is optional (its presence/absence was a real choice), OR this item (or a later
    one) is an unknown/extra key not in the spec at all (an anomaly, itself a
    choice). Computed purely from the spec + which keys are present -- not from
    "which optional keys happen to be present in this instance" -- so an instance
    with every optional key absent is weighted the same as one with some present
    (fixes the omission bug), and two required keys separated only by a *later*
    optional key still get a scaffold separator between them (fixes the
    multi-required-key bug)."""
    k, known = items[idx]
    if not known:
        return True
    pos = key_order.index(k)
    if any(x not in required for x in key_order[pos + 1 :]):
        return True
    if any(not kn for (_, kn) in items[idx + 1 :]):
        return True
    return False


def _write_obj_like(key_order, required, value, get_sub_spec, pieces):
    if not isinstance(value, dict):
        raise SchemaMismatch(f"expected object, got {type(value).__name__}")
    pieces.append(("{", SCAFFOLD))
    items = [(k, True) for k in key_order if k in value]
    items += [(k, False) for k in value if k not in key_order]
    n = len(items)
    if n == 0:
        has_optional = any(k not in required for k in key_order)
        pieces.append(("}", DECISION if has_optional else SCAFFOLD))
        return
    for idx, (k, known) in enumerate(items):
        pieces.append(('"', SCAFFOLD))
        # An optional key's mere presence is a decision; a required key's name
        # carries no information (it was always going to be there).
        pieces.append((k, SCAFFOLD if (known and k in required) else DECISION))
        pieces.append(('"', SCAFFOLD))
        pieces.append((": ", SCAFFOLD))
        if known:
            _write(get_sub_spec(k), value[k], pieces)
        else:
            _write_generic_value(value[k], pieces)
        cls = (
            DECISION
            if _trailing_has_choice(items, idx, key_order, required)
            else SCAFFOLD
        )
        pieces.append((", " if idx < n - 1 else "}", cls))


def _coupled_sub_spec(spec, k):
    # spec.tag/spec.value_key are plain discriminator/value strings with no node type
    # of their own (their legal sets are `spec.variants`, enforced by validate(), not
    # by the writer) -- weight them like an ordinary Str value: quotes scaffold,
    # content decision. Identical weighting to Enum, so Str() is an exact stand-in.
    return spec.keys[k] if k in spec.keys else Str()


def _write(spec, value, pieces):
    if isinstance(spec, Obj):
        _write_obj_like(
            tuple(spec.keys), spec.required, value, lambda k: spec.keys[k], pieces
        )
    elif isinstance(spec, DynObj):
        _write_obj_like(spec.key_vocab, (), value, lambda k: spec.value, pieces)
    elif isinstance(spec, CoupledObj):
        _write_obj_like(
            spec.key_order,
            spec.required,
            value,
            lambda k: _coupled_sub_spec(spec, k),
            pieces,
        )
    elif isinstance(spec, Arr):
        _write_arr(spec.item, value, pieces)
    elif isinstance(spec, Enum):
        _write_string_value(value, pieces, nullable=spec.nullable)
    elif isinstance(spec, Str):
        _write_string_value(value, pieces, nullable=spec.nullable)
    elif isinstance(spec, Num):
        _write_num_value(value, pieces, nullable=spec.nullable)
    else:
        _write_generic_value(value, pieces)


def _merge_spans(pieces):
    spans = []
    pos = 0
    cur_start, cur_cls = 0, None
    for text, cls in pieces:
        if cur_cls is None:
            cur_start = pos
        elif cls != cur_cls:
            spans.append((cur_start, pos, cur_cls))
            cur_start = pos
        cur_cls = cls
        pos += len(text)
    if cur_cls is not None:
        spans.append((cur_start, pos, cur_cls))
    return spans


class WeightedJSONWriter:
    """Serializes a canonicalized target back to JSON text alongside per-character
    decision/scaffold spans, using the same default `json.dumps` separators
    (`", "`/`": "`) meddata's own `_target_str()` uses -- so re-serializing
    already-canonical input is byte-identical to the source."""

    def serialize(self, task, target):
        canon = canonicalize(task, target)
        spec = SPECS.get(task)
        pieces = []
        if spec is None:
            _write_generic_value(canon, pieces)
        else:
            _write(spec, canon, pieces)
        text = "".join(t for t, _ in pieces)
        spans = _merge_spans(pieces)
        return text, spans


# --- chat-template integration -----------------------------------------------------

_FALLBACK_REASON_LOGGED = set()


def _log_fallback_once(task, reason):
    key = (task, reason)
    if key not in _FALLBACK_REASON_LOGGED:
        _FALLBACK_REASON_LOGGED.add(key)
        logger.warning(
            f"[weighted_json_loss] task={task!r} falling back to uniform (plain-CE) "
            f"weighting: {reason}"
        )


def WeightedJSON_apply_chat_template(
    example, tokenizer, enable_thinking=None, use_generation_prompt=False
):
    """Mirrors `PLW_apply_chat_template`'s prompt/completion split (same prefix trick
    via `extract_assistant_text`), but additionally re-serializes the assistant
    target through `WeightedJSONWriter` and returns per-character decision/scaffold
    spans located inside the rendered completion.

    On any failure (bad JSON, shape mismatch vs. the task's spec, or the canonical
    text not found exactly once in the rendered completion) falls back to uniform
    decision weight (1.0) everywhere -- never worse than plain CE for that example --
    and records `example["_wjl_fallback"] = True` so `report_fallback_rates` can
    compute an exact per-task fallback rate from the processed dataset itself (a
    module-level counter would not aggregate correctly across `num_proc` worker
    processes).
    """
    messages = example["messages"]
    task = example.get("task", "") or ""

    prompts = [m for m in messages if m["role"] != "assistant"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    fallback_reason = None
    canon_text, spans = None, None
    if not assistant_msgs:
        fallback_reason = "no assistant turn"
    else:
        try:
            obj = json.loads(assistant_msgs[-1]["content"])
            canon_text, spans = WeightedJSONWriter().serialize(task, obj)
        except (json.JSONDecodeError, SchemaMismatch, TypeError, ValueError) as e:
            fallback_reason = f"{type(e).__name__}: {e}"

    labels = [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
    if fallback_reason is None:
        labels += assistant_msgs[:-1] + [{**assistant_msgs[-1], "content": canon_text}]
    else:
        labels += assistant_msgs

    template_kwargs = (
        {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
    )
    example["prompt"] = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=use_generation_prompt,
        **template_kwargs,
    )
    completion_text = extract_assistant_text(
        tokenizer,
        labels,
        enable_thinking=enable_thinking,
        use_generation_prompt=use_generation_prompt,
    )
    example["completion"] = completion_text

    if fallback_reason is None:
        start = completion_text.find(canon_text)
        rstart = completion_text.rfind(canon_text)
        if start == -1 or start != rstart:
            fallback_reason = (
                "canonical text not found exactly once in rendered completion"
            )

    if fallback_reason is not None:
        _log_fallback_once(task, fallback_reason)
        example["completion_spans"] = [(0, len(completion_text), DECISION)]
        example["_wjl_fallback"] = True
    else:
        full_spans = []
        if start > 0:
            full_spans.append((0, start, SCAFFOLD))
        full_spans.extend((s + start, e + start, c) for (s, e, c) in spans)
        end = start + len(canon_text)
        if end < len(completion_text):
            full_spans.append((end, len(completion_text), SCAFFOLD))
        example["completion_spans"] = full_spans
        example["_wjl_fallback"] = False

    example["task"] = task
    return example


def report_fallback_rates(dataset_dict, threshold=0.01):
    """Compute per-task fallback rate from the `_wjl_fallback`/`task` columns a
    processed dataset (or DatasetDict) carries after `.map(WeightedJSON_apply_chat_template)`.
    Reading it back from the dataset (rather than a module-level counter incremented
    inside the map function) is what makes this correct under `num_proc > 1`, where
    map workers are separate processes and don't share Python state.

    Logs every task's rate and raises if any exceeds `threshold`."""
    splits = dataset_dict.values() if isinstance(dataset_dict, dict) else [dataset_dict]
    totals, fails = {}, {}
    for split in splits:
        if "_wjl_fallback" not in split.column_names:
            continue
        for task, flag in zip(split["task"], split["_wjl_fallback"]):
            totals[task] = totals.get(task, 0) + 1
            fails[task] = fails.get(task, 0) + int(flag)

    problems = []
    for task, total in totals.items():
        rate = fails.get(task, 0) / total if total else 0.0
        logger.info(
            f"[weighted_json_loss] task={task!r} fallback_rate={rate:.4f} "
            f"({fails.get(task, 0)}/{total})"
        )
        if rate > threshold:
            problems.append((task, rate, fails.get(task, 0), total))

    if problems:
        raise RuntimeError(
            f"weighted_json_loss fallback rate exceeded threshold {threshold:.2%} for "
            f"tasks: {problems}. This means the ported schema/writer disagrees with real "
            f"data for these tasks -- investigate before training on partially "
            f"uniform-weighted (i.e. plain-CE) examples."
        )
    return {task: (fails.get(task, 0), total) for task, total in totals.items()}


# --- offsets -> token weights --------------------------------------------------------


def token_weights_from_offsets(offsets, spans, scaffold_weight):
    """Map each token's `(char_start, char_end)` offset to a (weight, class) pair:
    the max class of any span it overlaps (max-over-covered-characters can only ever
    upgrade a token toward decision weight, never downgrade -- a token that overlaps
    even one decision character is itself a decision token). Empty-offset tokens
    (e.g. added/special tokens under some tokenizers) default to scaffold."""
    weights, classes = [], []
    for cs, ce in offsets:
        if ce <= cs:
            classes.append(SCAFFOLD)
            weights.append(scaffold_weight)
            continue
        cls = SCAFFOLD
        for s, e, c in spans:
            if s < ce and e > cs and c > cls:
                cls = c
        classes.append(cls)
        weights.append(1.0 if cls == DECISION else scaffold_weight)
    return weights, classes


def audit_weights(tokenizer, task, target_obj, scaffold_weight=0.1):
    """Debugging helper (no torch dependency): serializes `target_obj` for `task`,
    tokenizes with offsets, and prints each token with its class/weight."""
    canon_text, spans = WeightedJSONWriter().serialize(task, target_obj)
    enc = tokenizer(canon_text, add_special_tokens=False, return_offsets_mapping=True)
    weights, classes = token_weights_from_offsets(
        enc["offset_mapping"], spans, scaffold_weight
    )
    print(f"task={task!r}  text={canon_text!r}")
    for tid, (cs, ce), w, c in zip(
        enc["input_ids"], enc["offset_mapping"], weights, classes
    ):
        piece = canon_text[cs:ce]
        cls_name = "DECISION" if c == DECISION else "SCAFFOLD"
        print(f"  {tid:>8}  {piece!r:24}  {cls_name:8}  weight={w}")
    return weights, classes


# --- trainer -------------------------------------------------------------------------


@dataclass
class WeightedJSONCollator(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples):
        input_ids = [torch.tensor(e["input_ids"]) for e in examples]
        attention_mask = [torch.tensor(e["attention_mask"]) for e in examples]
        labels = [torch.tensor(e["labels"]) for e in examples]
        loss_weights = [
            torch.tensor(e["loss_weights"], dtype=torch.float) for e in examples
        ]
        decision_mask = [
            torch.tensor(e["decision_mask"], dtype=torch.bool) for e in examples
        ]
        task_ids = torch.tensor([e["task_id"] for e in examples], dtype=torch.long)

        output = {}
        output["input_ids"] = pad(
            input_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        output["attention_mask"] = pad(
            attention_mask, padding_value=0, padding_side="right"
        )
        output["labels"] = pad(labels, padding_value=-100, padding_side="right")
        output["loss_weights"] = pad(
            loss_weights, padding_value=0.0, padding_side="right"
        )
        output["decision_mask"] = pad(
            decision_mask, padding_value=False, padding_side="right"
        )
        output["task_ids"] = task_ids
        return output


class WeightedJSONTrainer(SFTTrainer):
    """Structural analog of `PLWTrainer` for per-token weighted-JSON loss instead of
    prompt/completion weighting: custom collator, `_set_signature_columns_if_needed`
    override (needed for the same `remove_unused_columns` reason PLW documents),
    `_prepare_dataset` override converting `completion_spans` -> per-token weights,
    and a `compute_loss` override doing `sum(w * CE) / sum(w)`.
    """

    def __init__(self, *args, **kwargs):
        args_obj = kwargs.get("args")
        if args_obj is not None:
            if getattr(args_obj, "packing", False):
                raise ValueError("WeightedJSONTrainer requires packing=False.")
            if getattr(args_obj, "padding_free", False):
                raise ValueError(
                    "WeightedJSONTrainer requires padding_free=False (padding-free mode's "
                    "variable-shape batching is incompatible with this collator's fixed "
                    "per-token loss_weights/decision_mask tensors)."
                )
            if getattr(args_obj, "use_liger_kernel", False):
                raise ValueError(
                    "WeightedJSONTrainer is incompatible with use_liger_kernel=True (the "
                    "liger fused CE kernel bypasses per-token loss weighting entirely)."
                )
            if getattr(args_obj, "gradient_accumulation_steps", 1) > 1:
                logger.warning(
                    "WeightedJSONTrainer's sum(w*CE)/sum(w) normalization is only exact "
                    "within a micro-batch; at gradient_accumulation_steps > 1 this can "
                    "differ from the true global weighted mean (same caveat PLWTrainer "
                    "documents for its own per-microbatch normalization)."
                )

        for name in ("_prepare_dataset", "_set_signature_columns_if_needed"):
            if not hasattr(SFTTrainer, name):
                raise RuntimeError(
                    f"Installed trl.SFTTrainer no longer defines {name!r} -- "
                    "WeightedJSONTrainer's override relies on this private method; "
                    "check the installed trl version."
                )

        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        train_dataset = kwargs.get("train_dataset")
        tasks = []
        if train_dataset is not None and "task" in getattr(
            train_dataset, "column_names", []
        ):
            tasks = sorted(set(train_dataset["task"]))
        self.task_vocab = tasks
        self.task_to_id = {t: i for i, t in enumerate(tasks)}

        data_collator = WeightedJSONCollator(pad_token_id=tokenizer.pad_token_id)
        self.loss_fct = CrossEntropyLoss(reduction="none")

        super().__init__(*args, data_collator=data_collator, **kwargs)

    def _set_signature_columns_if_needed(self):
        # Same rationale as PLWTrainer: remove_unused_columns would otherwise strip
        # every column compute_loss/the collator need before collation even runs.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "labels",
                "loss_weights",
                "decision_mask",
                "task_id",
            ]

    def _activation_checkpointing_active(self):
        if self.args.gradient_checkpointing:
            return True
        plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        return bool(
            plugin is not None and getattr(plugin, "activation_checkpointing", False)
        )

    def _prepare_dataset(
        self, dataset, processing_class, args, packing, formatting_func, dataset_name
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        column_names = (
            dataset.column_names
            if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
            else None
        )
        if column_names and "input_ids" in column_names:
            return dataset
        if isinstance(
            dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset)
        ) and not isinstance(dataset, datasets.IterableDataset):
            return dataset

        self._dataset_sanity_checked = False

        max_len = (
            getattr(args, "max_length", None)
            or getattr(args, "max_seq_length", None)
            or processing_class.model_max_length
        )
        strategy = getattr(args, "weighted_json_overlength_strategy", "drop")
        scaffold_weight = args.weighted_json_scaffold_weight
        task_to_id = self.task_to_id
        logger.info(
            f"WeightedJSON length cap: {max_len}, over-length strategy: {strategy}, "
            f"scaffold_weight: {scaffold_weight}"
        )

        def tokenize(element):
            prompts = element["prompt"]
            completions = element["completion"]
            spans_list = element["completion_spans"]
            tasks = element["task"]

            out_input_ids, out_attn, out_labels = [], [], []
            out_weights, out_decision, out_task_id = [], [], []

            for prmp, comp, spans, task in zip(prompts, completions, spans_list, tasks):
                p = processing_class(
                    prmp,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_length=True,
                )
                l = processing_class(
                    comp,
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_length=True,
                    return_offsets_mapping=True,
                )

                p_len = p["length"][0]
                l_len = l["length"][0]
                total_len = p_len + l_len
                task_id = task_to_id.get(task, -1)

                if total_len > max_len:
                    if strategy == "error":
                        raise ValueError(
                            f"over-length example ({total_len} > {max_len}); set "
                            "weighted_json_overlength_strategy to 'drop' or 'truncate_right'"
                        )
                    if strategy == "drop":
                        logger.warning(
                            f"dropping over-length example: {total_len} > {max_len}"
                        )
                        continue
                    if strategy == "truncate_right":
                        seq = (p["input_ids"] + l["input_ids"])[:max_len]
                        kept_p = min(p_len, max_len)
                        kept_l = len(seq) - kept_p
                        if kept_l == 0:
                            logger.warning(
                                f"truncate_right left no completion (p_len={p_len} >= "
                                f"{max_len}); dropping"
                            )
                            continue
                        comp_offsets = l["offset_mapping"][:kept_l]
                        w, cls = token_weights_from_offsets(
                            comp_offsets, spans, scaffold_weight
                        )
                        out_input_ids.append(seq)
                        out_attn.append([1] * len(seq))
                        out_labels.append([-100] * kept_p + l["input_ids"][:kept_l])
                        out_weights.append([0.0] * kept_p + w)
                        out_decision.append(
                            [False] * kept_p + [c == DECISION for c in cls]
                        )
                        out_task_id.append(task_id)
                        continue
                    raise ValueError(
                        f"unknown weighted_json_overlength_strategy: {strategy!r}"
                    )

                weights, cls = token_weights_from_offsets(
                    l["offset_mapping"], spans, scaffold_weight
                )
                if weights:
                    # The final completion token is the chat template's single stop
                    # token (e.g. Qwen's <|im_end|>) -- stopping is always a decision,
                    # forced to weight 1.0 regardless of what its (usually empty)
                    # offset mapped to. No separate EOS is appended anywhere in this
                    # pipeline, so this never double-counts a stop token.
                    weights[-1] = 1.0
                    cls[-1] = DECISION

                out_input_ids.append(p["input_ids"] + l["input_ids"])
                out_attn.append(p["attention_mask"] + l["attention_mask"])
                out_labels.append([-100] * p_len + l["input_ids"])
                out_weights.append([0.0] * p_len + weights)
                out_decision.append([False] * p_len + [c == DECISION for c in cls])
                out_task_id.append(task_id)

            return {
                "input_ids": out_input_ids,
                "attention_mask": out_attn,
                "labels": out_labels,
                "loss_weights": out_weights,
                "decision_mask": out_decision,
                "task_id": out_task_id,
            }

        before = len(dataset)
        with PartialState().main_process_first():
            dataset = dataset.map(
                tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=args.dataset_num_proc,
            )
        logger.info(
            f"WeightedJSON over-length handling removed/trimmed {before - len(dataset)} of "
            f"{before} examples (strategy={strategy}, cap={max_len})"
        )
        return dataset

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if self.model.training else "eval"

        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        decision_mask = inputs.pop("decision_mask")
        task_ids = inputs.pop("task_ids")

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

        # No chunking: PLWTrainer's compute_loss establishes (see its comment) that
        # chunking the CE call alone does not reduce peak memory, since autograd
        # retains the full logits input for backward regardless of how many python
        # calls compute it in. Single flat CE call, same as PLWTrainer.
        V = logits.size(-1)
        flat = logits[:, :-1, :].reshape(-1, V)
        shift_labels = labels[:, 1:].to(flat.device)
        shift_weights = loss_weights[:, 1:].to(flat.device, flat.dtype)
        shift_decision = decision_mask[:, 1:].to(flat.device)

        token_losses = self.loss_fct(flat, shift_labels.reshape(-1))

        accumulating = self.args.gradient_accumulation_steps > 1
        new_regime = bool(getattr(self, "model_accepts_loss_kwargs", False))
        if accumulating and new_regime and num_items_in_batch is not None:
            denom = num_items_in_batch
        else:
            denom = shift_weights.sum()
        loss = token_losses.float() @ shift_weights.reshape(-1).float() / (denom + 1e-9)

        self._metrics[mode]["loss_weighted"].append(loss.item())

        interval = getattr(self.args, "plw_accuracy_log_steps", 50)
        if not getattr(self.args, "use_liger_kernel", False) and (
            interval <= 1 or self.state.global_step % interval == 0 or mode == "eval"
        ):
            predictions = flat.argmax(dim=-1).view(logits.size(0), -1)
            # Metrics are restricted to decision-class tokens (not merely
            # loss_weights != 0), so a control run with scaffold_weight=1.0 -- where
            # every completion token has weight 1.0 -- still reports a decision-only
            # accuracy rather than collapsing back to plain per-token accuracy.
            mask = shift_decision
            correct = (predictions == shift_labels) & mask
            total_tokens = self.accelerator.gather_for_metrics(mask.sum())
            correct_tokens = self.accelerator.gather_for_metrics(correct.sum())
            total_sum = total_tokens.sum()
            accuracy = (
                (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            )
            self._metrics[mode]["decision_token_accuracy"].append(accuracy)

            shift_task_ids = (
                task_ids.unsqueeze(1).expand(-1, mask.size(1)).to(flat.device)
            )
            for task_name, tid in self.task_to_id.items():
                task_mask = mask & (shift_task_ids == tid)
                t_total = self.accelerator.gather_for_metrics(task_mask.sum())
                t_total_sum = t_total.sum()
                if t_total_sum.item() == 0:
                    continue
                t_correct = self.accelerator.gather_for_metrics(
                    ((predictions == shift_labels) & task_mask).sum()
                )
                self._metrics[mode][f"accuracy/{task_name}"].append(
                    (t_correct.sum() / t_total_sum).item()
                )

        return (loss, outputs) if return_outputs else loss


# --- self-check ----------------------------------------------------------------------

_SELF_CHECK_EXAMPLES = {
    "nli": {"label": "entailment"},
    "mcqa": {"answer": "C"},
    "sts_regression": {"similarity": 3.5},
    "multilabel_cls": {"labels": ["Treatment", "Diagnosis"]},
    "ner_re_medication": {
        "drugs": [
            {"text": "Lipitor", "Strength": ["10 mg"], "Reason": ["hyperlipidemia"]}
        ],
        "unlinked": [{"type": "Reason", "text": "hyperlipidemia"}],
    },
    "medication_event": {
        "events": [
            {
                "id": "E1",
                "type": "Disposition",
                "trigger_text": "started",
                "attributes": {"Action": "Start", "Certainty": "Certain"},
            }
        ],
    },
    "sdoh_event": {
        "events": [
            {
                "id": "E2",
                "type": "Tobacco",
                "trigger_text": "smokes",
                "roles": {"Status": {"type": "StatusTime", "text": "current"}},
                "attributes": [
                    {"role": "Status", "name": "StatusTimeVal", "value": "current"}
                ],
            }
        ],
    },
    "concept_assertion_relation": {
        "concepts": [
            {"id": "c0", "type": "problem", "text": "diabetes", "assertion": "present"}
        ],
        "relations": [{"type": "TrIP", "arg1": "c1", "arg2": "c0"}],
    },
    "family_history": {
        "living_status": [{"member": "Father", "side": "Paternal", "score": "0"}],
        "observations": [
            {
                "member": "Father",
                "side": "Paternal",
                "text": "heart disease",
                "negation": "Non_Negated",
            }
        ],
    },
}


def _self_check():
    ok = True
    for task, target in _SELF_CHECK_EXAMPLES.items():
        try:
            text, spans = WeightedJSONWriter().serialize(task, target)
            reparsed = json.loads(text)
            expected = canonicalize(task, target)
            match = reparsed == expected
            covered = spans and spans[-1][1] == len(text) and spans[0][0] == 0
            status = "OK" if match and covered else "FAIL"
        except Exception as e:
            match, covered, status = False, False, f"ERROR: {e}"
            text = ""
        if status != "OK":
            ok = False
        print(
            f"  {task:30s} round_trip={match}  span_coverage={bool(covered)}  [{status}]"
        )
    print("self-check:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _self_check()
