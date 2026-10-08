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
"""HRM-Text tokenizer: load + setup in one place.

Handles both cases:
  - A pre-built HRM tokenizer (already has HRM special tokens/chat_template)
    -> loaded as-is, validated, returned.
  - A stock Qwen3 tokenizer (e.g. ``Qwen/Qwen3-0.6B``)
    -> loaded, HRM overrides applied, validated, returned.

The single entry point is ``load_hrm_tokenizer(path_or_id, ...)``.
"""

import logging
from typing import Dict

from transformers import AutoTokenizer

from .data import BOQ_TOKEN, CONDITION_MAP, EOA_TOKEN, EOQ_TOKEN

logger = logging.getLogger(__name__)

# Pad token
PAD_TOKEN = "<|endoftext|>"

# Jinja2 chat_template that reproduces the native training format:
#   <|im_start|><condition_token(s)><instruction><|im_end|><response><|box_end|>
HRM_CHAT_TEMPLATE = (
    "{%- set cond_map = {"
    '"direct": "<|object_ref_start|>", '
    '"cot": "<|object_ref_end|>", '
    '"noisy": "<|quad_start|>", '
    '"synth": "<|quad_end|>"'
    "} -%}"
    '{%- set ns = namespace(cond_tokens="") -%}'
    '{%- set cond_str = condition | default("direct") -%}'
    '{%- for c in cond_str.split(",") -%}'
    "{%- set ns.cond_tokens = ns.cond_tokens + cond_map[c.strip()] -%}"
    "{%- endfor -%}"
    '{%- set ns_instr = namespace(instruction="") -%}'
    "{%- for message in messages -%}"
    '{%- if message["role"] == "system" -%}'
    '{%- set ns_instr.instruction = message["content"] + "\\n\\n" + ns_instr.instruction -%}'
    '{%- elif message["role"] == "user" -%}'
    '{%- set ns_instr.instruction = ns_instr.instruction + message["content"] -%}'
    '{%- elif message["role"] == "assistant" -%}'
    '{{ message["content"] }}<|box_end|>'
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "<|im_start|>{{ ns.cond_tokens }}{{ ns_instr.instruction }}<|im_end|>"
    "{%- endif -%}"
)


def _is_hrm_tokenizer(tokenizer) -> bool:
    """Check if a tokenizer already has HRM special token assignments."""
    return (
        getattr(tokenizer, "eos_token", None) == EOA_TOKEN
        and getattr(tokenizer, "bos_token", None) == BOQ_TOKEN
    )


def _apply_hrm_overrides(tokenizer) -> None:
    """Apply HRM-Text special token assignments (bos/eos/pad) in place."""
    tokenizer.bos_token = BOQ_TOKEN  # <|im_start|>
    tokenizer.eos_token = EOA_TOKEN  # <|box_end|>
    tokenizer.pad_token = PAD_TOKEN  # <|endoftext|>


def _validate_hrm_tokenizer(tokenizer) -> None:
    """Validate all required HRM special tokens resolve to real, present-in-vocab ids.

    Some checkpoints (e.g. sapientinc/HRM-Text-1B) set ``unk_token`` to the
    same string as ``pad_token`` ("<|endoftext|>"). For those, PAD's id
    trivially equals ``unk_token_id`` without PAD being missing from the
    vocab -- so a required token that IS the tokenizer's own ``unk_token``
    string is exempted from the UNK check. Genuinely missing tokens (not in
    vocab at all) are still caught via ``tok_id is None``, independent of
    whether ``unk_token_id`` happens to be set.
    """
    unk_id = tokenizer.unk_token_id
    unk_token_str = tokenizer.unk_token
    required: Dict[str, str] = {
        "BOQ": BOQ_TOKEN,
        "EOQ": EOQ_TOKEN,
        "EOA": EOA_TOKEN,
        "PAD": PAD_TOKEN,
        **{f"cond:{k}": v for k, v in CONDITION_MAP.items()},
    }
    for name, tok in required.items():
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None:
            raise ValueError(
                f"HRM special token '{tok}' ({name}) is not present in this tokenizer's vocab."
            )
        if tok != unk_token_str and unk_id is not None and tok_id == unk_id:
            raise ValueError(
                f"HRM special token '{tok}' ({name}) maps to UNK in this tokenizer. "
                f"Ensure you are using a Qwen3-based tokenizer that has these tokens in its vocab."
            )


def load_hrm_tokenizer(
    tokenizer_name_or_path: str,
    revision: str = "main",
    trust_remote_code: bool = True,
) -> "AutoTokenizer":
    """Load a tokenizer and ensure it has proper HRM-Text configuration.

    Handles all cases:
      - Pre-built HRM tokenizer (already has correct bos/eos/pad)
        -> loaded as-is; bos/eos/pad untouched.
      - Stock Qwen3 tokenizer (e.g. "Qwen/Qwen3-0.6B")
        -> loaded, HRM overrides applied (bos/eos/pad).
      - Any other tokenizer with the required tokens in vocab
        -> same as Qwen3 path.
    In every case, if the loaded tokenizer has no chat_template set, the HRM
    chat_template is installed -- a tokenizer can independently satisfy
    _is_hrm_tokenizer's bos/eos check (e.g. sapientinc/HRM-Text-1B's
    published config) while still lacking a chat_template, so this is not
    gated on which branch above ran. An existing chat_template is never
    overwritten.

    Args:
        tokenizer_name_or_path: HuggingFace Hub ID or local path.
        revision: Branch / tag / commit for Hub models.
        trust_remote_code: Pass through to AutoTokenizer.

    Returns:
        A fully configured HRM tokenizer ready for training.

    Raises:
        ValueError: if required HRM special tokens are not in the vocab.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if _is_hrm_tokenizer(tokenizer):
        logger.info(
            f"Loaded pre-built HRM tokenizer from: {tokenizer_name_or_path} "
            f"(bos={tokenizer.bos_token!r} eos={tokenizer.eos_token!r})"
        )
    else:
        logger.info(
            f"Loaded base tokenizer from: {tokenizer_name_or_path} -- "
            f"applying HRM special token overrides"
        )
        _apply_hrm_overrides(tokenizer)

    if tokenizer.chat_template is None:
        logger.info(
            "Tokenizer has no chat_template -- installing the HRM chat_template."
        )
        tokenizer.chat_template = HRM_CHAT_TEMPLATE

    _validate_hrm_tokenizer(tokenizer)

    logger.info(
        f"HRM tokenizer ready: bos={tokenizer.bos_token!r} eos={tokenizer.eos_token!r} "
        f"pad={tokenizer.pad_token!r} vocab_size={len(tokenizer)}"
    )
    return tokenizer
