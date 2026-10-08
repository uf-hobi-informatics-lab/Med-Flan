# coding=utf-8
# Copyright 2024 The Eli Lilly AADS Team. All rights reserved.
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
"""Argument dataclasses for HRM-Text SFT.

Deliberately separate from ``alignment.configs`` (top-level
``ModelArguments``/``DataArguments``/``SFTConfig``): HRM-Text needs a single
``dataset_path`` (not the multi-source ``dataset_mixer`` dict), a ``condition``
control-code field, and a fixed (non-warmup) truncated-backprop window
(``bp_steps``) that no other recipe in this repo uses. Reuse
``alignment.H4ArgumentParser`` for parsing -- it is generic and unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import TrainingArguments


VALID_BP_STEPS = (5, 6, 7, 8)


@dataclass
class ModelArguments:
    """Arguments specifying which HRM-Text model / tokenizer to load."""

    model_name_or_path: str = field(
        default="",
        metadata={
            "help": "HuggingFace Hub ID or local path to the HRM-Text checkpoint."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Branch / tag / commit to load from the Hub."},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Load model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Pass trust_remote_code=True when loading the HRM-Text model."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention backend, e.g. 'flash_attention_2'. HRM's PrefixLM mask is derived from token_type_ids independent of this choice (see DataCollatorForHRM)."
        },
    )
    condition: str = field(
        default="direct",
        metadata={
            "help": (
                "Comma-separated HRM condition(s) prepended to every prompt as a style "
                "control code. Choices per token: direct, cot, noisy, synth. "
                "Examples: 'direct', 'cot', 'synth,cot'."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer to load. Defaults to model_name_or_path when unset."
        },
    )
    # PEFT / quantization: kept for CLI parity with the reference script only.
    # Not wired to any loading logic -- LoRA has not been validated against
    # this recurrent architecture (reports from downstream fine-tunes describe
    # it as unstable), so full fine-tuning is the only supported path here.
    use_peft: bool = field(
        default=False,
        metadata={"help": "Not implemented for HRM-Text; kept for CLI parity only."},
    )
    lora_r: int = field(
        default=16, metadata={"help": "LoRA rank (unused -- see use_peft)."}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "LoRA alpha (unused -- see use_peft)."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout (unused -- see use_peft)."}
    )
    lora_target_modules: Union[List[str], str] = field(
        default="all",
        metadata={"help": "LoRA target modules (unused -- see use_peft)."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Not implemented for HRM-Text; kept for CLI parity only."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Not implemented for HRM-Text; kept for CLI parity only."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "Cannot use both 8-bit and 4-bit quantization simultaneously."
            )


@dataclass
class DataArguments:
    """Arguments specifying where HRM-Text training data lives and how to process it."""

    dataset_path: str = field(
        metadata={
            "help": "Path to the JSONL file or Hugging Face Arrow dataset directory."
        }
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the eval JSONL / Arrow dataset. Skipped if not set."
        },
    )
    max_seq_length: int = field(
        default=4096,
        metadata={
            "help": "Truncate sequences longer than this from the left (keeps the response/labels, trims the prompt prefix)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of parallel workers for dataset.map()."},
    )
    use_chat_template: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, build prompts via tokenizer.apply_chat_template() using the "
                "HRM chat_template. Expects a 'messages' key in each sample instead of a "
                "flat 'instruction'. Both paths produce identical token IDs for equivalent inputs."
            )
        },
    )
    dataset_splits: List[str] = field(
        default_factory=lambda: ["train"],
        metadata={
            "help": (
                "Split name(s) to read from dataset_path when it's an HF Arrow directory "
                "(dataset.save_to_disk()), e.g. ['train_sft', 'test_sft']. First element is "
                "the train split name (default 'train'). Second element, if given, is the "
                "eval split name -- it must exist or loading raises. If omitted, the eval "
                "split (if any) is auto-detected by trying 'test'/'validation'/'eval' in "
                "that order. Ignored for JSONL dataset_path (always a single 'train' split) "
                "and superseded by eval_dataset_path when that's also set."
            )
        },
    )


@dataclass
class SFTConfig(TrainingArguments):
    """Training arguments for HRM-Text SFT, extending HF TrainingArguments.

    Plain TrainingArguments (not trl.SFTConfig/SFTTrainer): samples are
    pre-tokenized by ``format_hrm_sample`` (which also produces the
    ``token_type_ids`` PrefixLM needs) and fed through a vanilla ``Trainer``
    with ``DataCollatorForHRM``.
    """

    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "Hub branch to push the saved model to (optional)."},
    )
    logging_first_step: bool = field(default=True)

    prompt_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": "Loss weight applied to instruction (prefix) tokens. 0 = fully masked."
        },
    )

    use_weighted_json_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Use per-token weighted cross-entropy on JSON-formatted targets "
                "(alignment.weighted_json_loss / alignment.hrm.format_hrm_weighted_json_sample) "
                "instead of plain CE. Mutually exclusive with prompt_loss_weight != 0."
            )
        },
    )
    weighted_json_scaffold_weight: float = field(
        default=0.1,
        metadata={
            "help": "Loss weight for SCAFFOLD-classified tokens (mandatory JSON punctuation/"
            "required-key boilerplate) when use_weighted_json_loss=True. DECISION tokens "
            "always get weight 1.0. Set to 1.0 for the canonicalization-only control run."
        },
    )

    bp_steps: int = field(
        default=5,
        metadata={
            "help": (
                "Fixed truncated-backprop window (K) for fine-tuning -- set once before "
                "training and held constant (no warmup schedule, unlike pretraining's "
                "bp_min_steps/bp_max_steps/bp_warmup_ratio). Applied via "
                "alignment.hrm.bp_cycles.apply_fixed_bp_cycles(model, bp_steps). Valid: "
                "5 (pretraining's terminal value) through 8 (full backprop through every "
                "H/L cycle for the shipped 1B config's H_cycles=2, L_cycles=3)."
            ),
            "choices": list(VALID_BP_STEPS),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.bp_steps not in VALID_BP_STEPS:
            raise ValueError(
                f"bp_steps must be one of {VALID_BP_STEPS}, got {self.bp_steps}."
            )
