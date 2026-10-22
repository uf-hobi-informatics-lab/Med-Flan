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
"""Sync a checkpoint's RoPE context window to a target sequence length.

Single rule, used by every SFT entry point in this repo (``scripts/run_sft.py``
for Ouro/Huginn/Llama/Qwen-family checkpoints, ``scripts/run_hf_sft.py`` for
HRM-Text): if the checkpoint's own trained context window is already >= the
recipe's target sequence length, do nothing. If it's smaller, extend it by
overriding the relevant config field(s) before the model is constructed --
no manual factor/method knobs to set.

Two RoPE implementation shapes are handled:
  - Standard dynamic RoPE (Llama/Qwen/Ouro/HRM-Text-style): position-dependent
    cos/sin computed per forward from ``position_ids``; ``max_position_embeddings``
    is a free config value, not baked into trained weights. Extending it is a
    safe override, paired with a scaled RoPE (``rope_scaling`` or the newer
    ``rope_parameters`` config attribute, whichever the checkpoint's config
    class exposes) instead of raw extrapolation.
  - Huginn/Raven-style: ``freqs_cis`` is a fixed-size buffer precomputed once
    at ``__init__`` from ``config.block_size``/``config.rope_base``. HF's
    ``PretrainedConfig.from_dict`` applies any kwarg matching an existing
    config attribute *before* the model class is constructed, so passing
    ``block_size``/``rope_base`` overrides through ``from_pretrained`` takes
    effect before ``_precompute_freqs_cis()`` runs -- no post-load buffer
    patch needed.
"""

import logging

from transformers import AutoConfig


logger = logging.getLogger(__name__)

DEFAULT_ROPE_SCALING_TYPE = "yarn"


def resolve_checkpoint_max_length(model_config) -> int:
    """Read a checkpoint's own trained max sequence length off its config.

    Checks, in order: ``max_position_embeddings``, ``text_config.
    max_position_embeddings`` (multimodal/composite configs), ``block_size``
    (Huginn/Raven-style configs, whose ``freqs_cis`` buffer is precomputed to
    exactly this length).
    """
    if hasattr(model_config, "max_position_embeddings"):
        return model_config.max_position_embeddings
    if hasattr(model_config, "text_config") and hasattr(
        model_config.text_config, "max_position_embeddings"
    ):
        return model_config.text_config.max_position_embeddings
    if hasattr(model_config, "block_size"):
        return model_config.block_size
    raise AttributeError(
        f"Could not find a max sequence length field on {type(model_config).__name__} "
        "-- expected one of `max_position_embeddings`, `text_config."
        "max_position_embeddings`, or `block_size`."
    )


def get_context_extension_kwargs(
    model_name_or_path: str,
    target_max_length: int,
    revision: str = "main",
    trust_remote_code: bool = True,
    rope_scaling_type: str = DEFAULT_ROPE_SCALING_TYPE,
) -> dict:
    """Kwargs to merge into ``model_kwargs`` so this checkpoint's context window
    covers ``target_max_length``.

    Returns ``{}`` (no-op) when the checkpoint's own trained max length is
    already >= ``target_max_length``, or when ``target_max_length`` is falsy.
    The returned kwargs are meant to be passed straight through to
    ``AutoModelForCausalLM.from_pretrained(..., **model_kwargs)`` (or TRL's
    ``model_init_kwargs``) -- they rely on HF's standard from_pretrained
    behavior of applying any kwarg matching a config attribute to the config
    before the model class is constructed.
    """
    if not target_max_length:
        return {}

    config = AutoConfig.from_pretrained(
        model_name_or_path, revision=revision, trust_remote_code=trust_remote_code
    )
    original_max_length = resolve_checkpoint_max_length(config)

    if target_max_length <= original_max_length:
        return {}

    factor = target_max_length / original_max_length
    logger.info(
        f"Extending context window for {model_name_or_path}: "
        f"{original_max_length} -> {target_max_length} (factor={factor:.3f})"
    )

    if hasattr(config, "block_size"):
        # Huginn/Raven: no rope_scaling/rope_parameters machinery -- just a plain
        # theta. Scale rope_base by the same factor as a simple NTK-style extension
        # instead of raw extrapolation.
        kwargs = {"block_size": target_max_length}
        if hasattr(config, "rope_base"):
            kwargs["rope_base"] = config.rope_base * factor
        return kwargs

    scaling_dict = {
        "factor": factor,
        "original_max_position_embeddings": original_max_length,
    }
    kwargs = {"max_position_embeddings": target_max_length}
    if hasattr(config, "rope_parameters"):
        # Newer transformers configs (e.g. HRM-Text) dispatch RoPE scaling via a
        # "rope_type" key on rope_parameters instead of the older rope_scaling dict --
        # and, critically, bundle rope_theta (the RoPE base) *inside* this dict rather
        # than as a separate top-level config attribute. Merge into the existing dict
        # instead of replacing it outright, or rope_theta gets dropped and
        # _compute_yarn_parameters crashes on `None ** tensor`.
        original_rope_parameters = config.rope_parameters or {}
        kwargs["rope_parameters"] = {
            **original_rope_parameters,
            "rope_type": rope_scaling_type,
            **scaling_dict,
        }
    else:
        original_rope_scaling = getattr(config, "rope_scaling", None) or {}
        kwargs["rope_scaling"] = {
            **original_rope_scaling,
            "type": rope_scaling_type,
            **scaling_dict,
        }
    return kwargs
