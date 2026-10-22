# coding=utf-8
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
"""HRM-Text data loading and prompt formatting.

Every sample is formatted as:

    <|im_start|><condition_token><instruction><|im_end|><response><|box_end|>
    |<--------------- prefix (bidirectional PrefixLM) ---------------->|
                                                        |<--- causal -->|

This matches HRM-Text's native training pipeline exactly. There are no
ChatML role headers ("system\\n", "user\\n") -- <|im_start|> and <|im_end|>
are structural delimiters only. The condition token is placed directly after
<|im_start|>.

Two formatting paths are available and must produce **identical token IDs**:

1. Manual path (``use_chat_template=False``, default):
   Builds the prompt string directly from ``example["instruction"]``.
   Used when the dataset already contains a flat, fully-constructed instruction.

2. chat_template path (``use_chat_template=True``):
   Uses ``tokenizer.apply_chat_template()`` with an OpenAI-style messages list.
   Requires the tokenizer to have the HRM chat_template installed (see
   ``alignment.hrm.tokenizer.load_hrm_tokenizer``).
   Converts the messages to the same flat format as path 1.

The HF model class consumes token_type_ids to reproduce the PrefixLM attention
pattern (1 = prefix / bidirectional, 0 = causal).

Input JSONL schema (manual path)
---------------------------------
Required keys:
  instruction  (str)  - the full prompt text, already constructed by the caller
  response     (str)  - the expected model output

Optional keys:
  condition    (str)  - one of "direct", "cot", "noisy", "synth" (comma-sep
                         for composition, e.g. "synth,cot"). Falls back to
                         model_args.condition when absent.

Input JSONL schema (chat_template path)
---------------------------------------
Required keys:
  messages     (List[dict])  - OpenAI message list with role/content dicts.
                                Supported roles: "system", "user", "assistant".
                                System content is prepended to user content
                                with "\\n\\n". If the list ends in an
                                "assistant" turn and there is no separate
                                "response" key, that trailing turn is used as
                                the response and excluded from the prompt --
                                standard OpenAI fine-tuning JSONL shape.

Optional keys:
  response     (str)  - explicit response text. Takes precedence over a
                         trailing assistant turn in "messages" when present.
  condition    (str)  - same as manual path
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HRM special-token vocabulary
# ---------------------------------------------------------------------------

#: Maps logical condition names to the special token strings used as prefixes.
CONDITION_MAP: Dict[str, str] = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}

#: Begin-of-query token (wraps the instruction)
BOQ_TOKEN = "<|im_start|>"
#: End-of-query token (closes the instruction prefix)
EOQ_TOKEN = "<|im_end|>"
#: End-of-answer token (generation stop token)
EOA_TOKEN = "<|box_end|>"

#: Generous upper bound on characters-per-token used to cap raw text before
#: tokenizing. Real text tokenizes at ~3-4 chars/token; this bound is only
#: meant to catch pathologically oversized rows (e.g. corrupted/mis-joined
#: fields) that would otherwise force the tokenizer to process an unbounded
#: string before any token-count-based truncation ever kicks in -- on a
#: large dataset processed via non-batched Dataset.map(num_proc=N), a single
#: such row can stall one worker for a long time while every other worker
#: sits idle.
_CHARS_PER_TOKEN_CAP = 8


def _build_condition_str(condition: str) -> str:
    """Return the concatenated condition token string for a (possibly composite) condition."""
    parts = [c.strip() for c in condition.split(",") if c.strip()]
    unknown = [p for p in parts if p not in CONDITION_MAP]
    if unknown:
        raise ValueError(
            f"Unknown condition(s): {unknown}. Valid choices: {list(CONDITION_MAP)}."
        )
    return "".join(CONDITION_MAP[p] for p in parts)


# ---------------------------------------------------------------------------
# Per-sample formatting helpers
# ---------------------------------------------------------------------------


def _build_prompt_str_manual(example: Dict, condition: str) -> str:
    """Build the HRM prompt string from a flat instruction field (native format)."""
    cond = example.get("condition", condition) or condition
    cond_str = _build_condition_str(cond)
    return f"{BOQ_TOKEN}{cond_str}{example['instruction']}{EOQ_TOKEN}"


def _split_chat_messages_and_response(example: Dict):
    """Split an OpenAI-style example into prompt-only messages + response text.

    Supports two input shapes for the chat_template path:
      - {"messages": [...prompt turns..., {"role": "assistant", "content": ...}]}
        (no separate "response" key) -- the trailing assistant turn is taken
        as the response; every turn before it is the prompt. This matches
        standard OpenAI fine-tuning JSONL, where the target output is the
        last message rather than a separate field.
      - {"messages": [...prompt turns only...], "response": "..."} -- explicit
        separate response field (also supported, e.g. when the same "response"
        column is shared with the manual/non-chat-template path).

    Returns (prompt_messages, response_text).
    """
    messages = example.get("messages")
    if "response" in example:
        return messages, example["response"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            "use_chat_template=True requires either a 'response' key or a 'messages' "
            "list ending in an assistant turn to use as the response."
        )
    return messages[:-1], messages[-1]["content"]


def _build_prompt_str_chat_template(example: Dict, tokenizer, condition: str) -> str:
    """Build the HRM prompt string via tokenizer.apply_chat_template().

    Converts an OpenAI-style messages list to the flat HRM format using the
    chat_template installed by ``alignment.hrm.tokenizer.load_hrm_tokenizer``.
    The output must be identical to ``_build_prompt_str_manual`` for
    equivalent single-turn inputs.

    Supports two input shapes:
      - ``example["messages"]``: list of role/content dicts (OpenAI format;
        should already exclude the trailing assistant/response turn -- see
        ``_split_chat_messages_and_response``)
      - ``example["instruction"]``: falls back to manual path

    The ``condition`` kwarg is forwarded to the template so the correct
    condition token is injected.
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "use_chat_template=True requires a tokenizer with a chat_template. "
            "Load the tokenizer via alignment.hrm.tokenizer.load_hrm_tokenizer."
        )

    cond = example.get("condition", condition) or condition

    if example.get("messages"):
        messages = example["messages"]
    else:
        # No messages list -- wrap the flat instruction as a user message
        messages = [{"role": "user", "content": example["instruction"]}]

    return tokenizer.apply_chat_template(
        messages,
        condition=cond,
        add_generation_prompt=True,
        tokenize=False,
    )


def format_hrm_sample(
    example: Dict,
    tokenizer,
    condition: str,
    max_seq_length: int,
    prompt_loss_weight: float = 0.0,
    use_chat_template: bool = False,
) -> Dict:
    """Format a single JSONL sample for SFT training.

    Args:
        example:            A dict with at minimum "instruction" and "response"
                            keys (manual path), or "messages" (chat_template
                            path -- either ending in a trailing assistant turn
                            used as the response, or paired with an explicit
                            "response" key).
        tokenizer:          HuggingFace tokenizer (must have chat_template set
                            when use_chat_template=True).
        condition:          Default condition ("direct", "cot", etc.). Per-sample
                            condition in example["condition"] takes precedence.
        max_seq_length:     Maximum token length. Samples longer than this are
                            left-truncated: the response (incl. its trailing
                            EOA stop token) is kept and the prompt prefix is
                            trimmed from the front, so labels in the final turn
                            are never lost. Matches truncation_side="left".
        prompt_loss_weight: Loss weight on instruction tokens. 0.0 = no loss
                            on the instruction (equivalent to native target_only=True).
        use_chat_template:  When True, build the prompt via
                            tokenizer.apply_chat_template() instead of the manual
                            format string. Both paths produce identical token IDs
                            for equivalent single-turn inputs.

    Returns a dict with:
        input_ids        (List[int])  - full token sequence
        labels           (List[int])  - -100 for prompt tokens (or actual IDs
                                        when prompt_loss_weight > 0)
        token_type_ids   (List[int])  - 1 for prefix, 0 for causal region
        attention_mask   (List[int])  - 1 for all real tokens
        prefix_len       (int)        - length of the prompt in tokens
    """
    if use_chat_template:
        prompt_messages, response_text = _split_chat_messages_and_response(example)
        prompt_str = _build_prompt_str_chat_template(
            {**example, "messages": prompt_messages}, tokenizer, condition
        )
    else:
        prompt_str = _build_prompt_str_manual(example, condition)
        response_text = example["response"]

    char_cap = max_seq_length * _CHARS_PER_TOKEN_CAP
    if len(prompt_str) > char_cap:
        prompt_str = prompt_str[:char_cap] + EOQ_TOKEN

    prompt_ids: List[int] = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

    # Build the response string (appended with EOA stop token)
    eoa_id: int = tokenizer.convert_tokens_to_ids(EOA_TOKEN)
    if len(response_text) > char_cap:
        response_text = response_text[:char_cap]
    resp_ids: List[int] = tokenizer(response_text, add_special_tokens=False)[
        "input_ids"
    ] + [eoa_id]

    # Left-truncate if the combined length exceeds max_seq_length: keep the
    # LAST max_seq_length tokens so the response (the causal region, where the
    # labels live) and its trailing EOA stop token survive, and the prompt
    # prefix is trimmed from the front. This matches the repo-wide
    # truncation_side="left" convention (run_sft.py / run_dpo.py / run_grpo.py /
    # ...: "Truncate from left to ensure we don't lose labels in final turn") --
    # the opposite of the old keep-full-prompt / drop-response-tail behavior,
    # which discarded exactly the tokens we train on whenever a row was long.
    total = len(prompt_ids) + len(resp_ids)
    if total > max_seq_length:
        if len(resp_ids) >= max_seq_length:
            # Response alone fills (or exceeds) the budget: keep its tail
            # (incl. EOA) and drop the prompt entirely -> prefix_len 0 (causal).
            resp_ids = resp_ids[-max_seq_length:]
            prompt_ids = []
        else:
            # Keep the full response + the tail of the prompt.
            keep_prompt = max_seq_length - len(resp_ids)
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + resp_ids
    prefix_len = len(prompt_ids)

    # Labels: mask instruction tokens (or apply prompt_loss_weight via a
    # separate mechanism in the collator / trainer). Store -100 for
    # instruction tokens when prompt_loss_weight == 0.
    if prompt_loss_weight == 0.0:
        labels = [-100] * prefix_len + list(resp_ids)
    else:
        # Shallow supervision on prompt: keep actual token IDs so the
        # DataCollatorForHRM can scale them by prompt_loss_weight.
        labels = list(prompt_ids) + list(resp_ids)

    # token_type_ids: 1 = bidirectional prefix, 0 = causal response
    token_type_ids = [1] * prefix_len + [0] * len(resp_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "prefix_len": prefix_len,
    }


def format_hrm_weighted_json_sample(
    example: Dict,
    tokenizer,
    condition: str,
    max_seq_length: int,
    scaffold_weight: float = 0.1,
) -> Dict:
    """Chat-template-only sibling of ``format_hrm_sample`` for per-token weighted-JSON
    loss (see ``alignment.weighted_json_loss``). Reuses that module's
    ``WeightedJSONWriter``/``token_weights_from_offsets`` to classify every response
    character as SCAFFOLD/DECISION, then tokenizes exactly like ``format_hrm_sample``'s
    chat_template path (same prompt-building helper, same left-truncation-on-overlength
    policy -- kept as HRM's one behavior rather than adding a
    ``weighted_json_overlength_strategy`` knob) with per-token weights/decision-mask
    carried alongside ``input_ids``/``labels``.

    Unlike the trl/qwen/ouro path (``WeightedJSON_apply_chat_template``), HRM's
    response text is tokenized standalone (not rendered through a chat template with
    surrounding wrapper text), so the writer's character spans -- offsets into the
    canonical JSON string -- line up directly with this function's own
    ``tokenizer(canon_text, return_offsets_mapping=True)`` call. No "locate the
    canonical text inside a larger rendered string" step is needed.

    Always takes the chat_template path (requires example["messages"]) since
    weighted-JSON loss only applies to the JSON-target med_flan data.
    """
    from ..weighted_json_loss import (
        DECISION,
        SchemaMismatch,
        WeightedJSONWriter,
        token_weights_from_offsets,
    )

    task = example.get("task", "") or ""
    prompt_messages, response_text = _split_chat_messages_and_response(example)

    fallback_reason = None
    canon_text, spans = response_text, None
    try:
        canon_text, spans = WeightedJSONWriter().serialize(
            task, json.loads(response_text)
        )
    except (json.JSONDecodeError, SchemaMismatch, TypeError, ValueError) as e:
        fallback_reason = f"{type(e).__name__}: {e}"

    prompt_str = _build_prompt_str_chat_template(
        {**example, "messages": prompt_messages}, tokenizer, condition
    )

    char_cap = max_seq_length * _CHARS_PER_TOKEN_CAP
    if len(prompt_str) > char_cap:
        prompt_str = prompt_str[:char_cap] + EOQ_TOKEN
    if len(canon_text) > char_cap:
        canon_text = canon_text[:char_cap]
        spans = None  # spans no longer line up with the truncated text; fall back
        fallback_reason = fallback_reason or "response exceeded char_cap"

    prompt_ids: List[int] = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

    eoa_id: int = tokenizer.convert_tokens_to_ids(EOA_TOKEN)
    enc = tokenizer(canon_text, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids: List[int] = enc["input_ids"] + [eoa_id]

    if spans is not None:
        weights, cls = token_weights_from_offsets(
            enc["offset_mapping"], spans, scaffold_weight
        )
    else:
        weights, cls = [1.0] * len(enc["input_ids"]), [DECISION] * len(enc["input_ids"])
    # The trailing EOA stop token is always a decision (stopping is a real choice).
    weights.append(1.0)
    cls.append(DECISION)

    # Left-truncate exactly like format_hrm_sample, slicing weights/cls in lockstep.
    total = len(prompt_ids) + len(resp_ids)
    if total > max_seq_length:
        if len(resp_ids) >= max_seq_length:
            resp_ids = resp_ids[-max_seq_length:]
            weights = weights[-max_seq_length:]
            cls = cls[-max_seq_length:]
            prompt_ids = []
        else:
            keep_prompt = max_seq_length - len(resp_ids)
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + resp_ids
    prefix_len = len(prompt_ids)

    return {
        "input_ids": input_ids,
        "labels": [-100] * prefix_len + list(resp_ids),
        "token_type_ids": [1] * prefix_len + [0] * len(resp_ids),
        "attention_mask": [1] * len(input_ids),
        "loss_weights": [0.0] * prefix_len + weights,
        "decision_mask": [False] * prefix_len + [c == DECISION for c in cls],
        "task": task,
        "_wjl_fallback": fallback_reason is not None,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str, num_proc: Optional[int] = None) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset, parsing in parallel across num_proc workers."""
    return load_dataset("json", data_files=path, num_proc=num_proc, split="train")


#: Split names checked (in order) when an Arrow DatasetDict is expected to
#: carry its own eval split alongside "train".
_EVAL_SPLIT_NAMES = ("test", "validation", "eval")


def _find_eval_split(dataset_dict: DatasetDict):
    for name in _EVAL_SPLIT_NAMES:
        if name in dataset_dict:
            return dataset_dict[name]
    return None


def get_datasets(
    dataset_path: str,
    eval_dataset_path: Optional[str] = None,
    num_proc: Optional[int] = None,
    dataset_splits: Optional[List[str]] = None,
) -> DatasetDict:
    """Load train (and optionally eval) datasets from JSONL or Arrow format.

    Supports:
      * A single .jsonl file - loaded as the train split.
      * A directory containing HuggingFace Arrow shards (saved with
        dataset.save_to_disk()) - loaded directly. If the directory is a
        DatasetDict with both a train split and an eval split, both are
        picked up from that single directory -- eval_dataset_path is not
        required in that case.

    dataset_splits controls which split names are read from an Arrow
    directory: dataset_splits[0] is the train split name (default "train");
    dataset_splits[1], if given, is the eval split name and must exist in
    the directory or loading raises. If dataset_splits has no second
    element, the eval split (if any) is auto-detected by trying
    "test"/"validation"/"eval" in that order. Ignored for JSONL dataset_path,
    which is always loaded as a single "train" split.

    eval_dataset_path, when given, always takes precedence over an eval
    split found alongside train in dataset_path.

    num_proc parallelizes JSONL parsing (ignored for Arrow shards, which load
    lazily via memory-mapping regardless).

    Returns a DatasetDict with a "train" key and optionally a "test" key.
    """
    p = Path(dataset_path)
    eval_ds = None
    train_split_name = dataset_splits[0] if dataset_splits else "train"
    eval_split_name = (
        dataset_splits[1] if dataset_splits and len(dataset_splits) > 1 else None
    )

    if p.is_file() and p.suffix in {".jsonl", ".json"}:
        logger.info(f"Loading train dataset from JSONL: {p}")
        train_ds = _load_jsonl(str(p), num_proc=num_proc)
    elif p.is_dir():
        logger.info(f"Loading train dataset from Arrow shards: {p}")
        loaded = load_from_disk(str(p))
        # load_from_disk may return a DatasetDict; unwrap if needed
        if isinstance(loaded, DatasetDict):
            if train_split_name not in loaded:
                raise ValueError(
                    f"Arrow dataset at {p} has no '{train_split_name}' split: {list(loaded.keys())}"
                )
            train_ds = loaded[train_split_name]
            if eval_dataset_path is None:
                if eval_split_name is not None:
                    if eval_split_name not in loaded:
                        raise ValueError(
                            f"Arrow dataset at {p} has no '{eval_split_name}' eval split: {list(loaded.keys())}"
                        )
                    eval_ds = loaded[eval_split_name]
                    logger.info(
                        f"Using configured eval split '{eval_split_name}' from {p}"
                    )
                else:
                    eval_ds = _find_eval_split(loaded)
                    if eval_ds is not None:
                        logger.info(
                            f"Found eval split alongside '{train_split_name}' in {p}"
                        )
        else:
            train_ds = loaded
    else:
        raise ValueError(
            f"dataset_path must be a .jsonl file or an Arrow directory, got: {p}"
        )

    result = DatasetDict({"train": train_ds})

    if eval_dataset_path is not None:
        ep = Path(eval_dataset_path)
        if ep.is_file() and ep.suffix in {".jsonl", ".json"}:
            logger.info(f"Loading eval dataset from JSONL: {ep}")
            eval_ds = _load_jsonl(str(ep), num_proc=num_proc)
        elif ep.is_dir():
            logger.info(f"Loading eval dataset from Arrow shards: {ep}")
            loaded_eval = load_from_disk(str(ep))
            if isinstance(loaded_eval, DatasetDict):
                eval_ds = _find_eval_split(loaded_eval)
                if eval_ds is None and "train" in loaded_eval:
                    eval_ds = loaded_eval["train"]
                if eval_ds is None:
                    raise ValueError(
                        f"Could not find test/validation/eval split in {ep}"
                    )
            else:
                eval_ds = loaded_eval
        else:
            raise ValueError(f"eval_dataset_path not found or unsupported format: {ep}")

    if eval_ds is not None:
        result["test"] = eval_ds

    return result
