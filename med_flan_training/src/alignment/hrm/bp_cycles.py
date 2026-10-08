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
"""Fixed (non-warmup) truncated-backprop window for HRM-Text SFT.

HRM-Text's pretraining pipeline linearly warms up the truncated-backprop
window K every step (``bp_min_steps`` -> ``bp_max_steps`` over
``bp_warmup_ratio`` of training, driven by a per-step ``TrainerCallback``).
Fine-tuning does not need that ramp -- this module ports just the pure
computation (``compute_l_bp_cycles``, unchanged from HRM-Text's
``hf_alignment/pretrain_trainer.py``) plus a one-time setter
(``apply_fixed_bp_cycles``) that applies a single fixed value once, before
training starts.
"""

import logging

logger = logging.getLogger(__name__)


def _unwrap_model(model):
    """Peel off DDP / DeepSpeed / Accelerate wrapper layers to reach the base model."""
    while hasattr(model, "module"):
        model = model.module
    return model


def compute_l_bp_cycles(H_cycles: int, L_cycles: int, bp_steps: int) -> list:
    """Per-H-cycle count of trailing L-iterations that receive gradient.

    Mirrors HRM-Text's ``conversion/convert_to_hf.py::_compute_l_bp_steps`` /
    ``hf_alignment/pretrain_trainer.py::compute_l_bp_cycles`` so a fixed
    ``bp_steps`` value reproduces the same truncated-backprop shape as the
    native pretraining pipeline and checkpoint converter.
    """
    h_bp_steps = min(H_cycles, max(0, bp_steps - 1))
    l_bp_steps = min(H_cycles * L_cycles, max(0, bp_steps - h_bp_steps))
    threshold = H_cycles * L_cycles - l_bp_steps
    return [
        max(0, min(L_cycles, (i + 1) * L_cycles - threshold)) for i in range(H_cycles)
    ]


def _find_hrm_text_model(model):
    """Locate the underlying HrmTextModel instance, wherever it lives.

    Works for both:
      - a standalone HrmTextForCausalLM (attribute path: model.model)
      - a Qwen3HrmForCausalLM, which nests HrmTextModel one level deeper as
        model.model.hrm

    Returns None if no HrmTextModel-like submodule (one exposing
    L_bp_cycles_padded) can be found, so callers can no-op gracefully on any
    model type that doesn't use HRM's truncated-backprop scheme.
    """
    base_model = _unwrap_model(model)
    candidate = getattr(base_model, "model", None)
    if candidate is not None and hasattr(candidate, "L_bp_cycles_padded"):
        return candidate
    nested = getattr(candidate, "hrm", None)
    if nested is not None and hasattr(nested, "L_bp_cycles_padded"):
        return nested
    return None


def apply_fixed_bp_cycles(model, bp_steps: int) -> None:
    """Set a fixed truncated-backprop window on ``model`` once, before training.

    No callback, no per-step recompute -- ``bp_steps`` is held constant for
    the whole fine-tuning run (unlike pretraining's linear warmup). No-ops
    (with a debug log) if ``model`` has no HRM submodule exposing
    ``L_bp_cycles_padded``.
    """
    hrm_model = _find_hrm_text_model(model)
    if hrm_model is None:
        logger.debug(
            "apply_fixed_bp_cycles: no HRM submodule found on model; skipping."
        )
        return

    config = hrm_model.config
    l_bp_cycles = compute_l_bp_cycles(config.H_cycles, config.L_cycles, bp_steps)
    hrm_model.L_bp_cycles_padded = l_bp_cycles
    logger.info(f"apply_fixed_bp_cycles: bp_steps={bp_steps} L_bp_cycles={l_bp_cycles}")
