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
"""HRM-Text SFT support.

Isolated from the top-level ``alignment`` package -- ``alignment.hrm``'s
``ModelArguments``/``DataArguments``/``SFTConfig`` are HRM-Text-specific and
distinct from ``alignment.ModelArguments``/etc. used by every other recipe.
Generic pieces (``H4ArgumentParser``, callbacks, ``get_checkpoint``, ...) are
reused directly from the top-level ``alignment`` package instead of being
duplicated here.
"""

from .bp_cycles import apply_fixed_bp_cycles, compute_l_bp_cycles
from .collator import (
    IGNORE_LABEL_ID,
    DataCollatorForHRM,
    DataCollatorForHRMWeightedJSON,
)
from .configs import DataArguments, ModelArguments, SFTConfig
from .data import format_hrm_sample, format_hrm_weighted_json_sample, get_datasets
from .tokenizer import load_hrm_tokenizer
from .trainer import HRMSFTTrainer, HRMWeightedJSONTrainer

__all__ = [
    "ModelArguments",
    "DataArguments",
    "SFTConfig",
    "DataCollatorForHRM",
    "DataCollatorForHRMWeightedJSON",
    "IGNORE_LABEL_ID",
    "format_hrm_sample",
    "format_hrm_weighted_json_sample",
    "get_datasets",
    "load_hrm_tokenizer",
    "apply_fixed_bp_cycles",
    "compute_l_bp_cycles",
    "HRMSFTTrainer",
    "HRMWeightedJSONTrainer",
]
