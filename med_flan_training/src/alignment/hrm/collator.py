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
"""Data collator for HRM-Text SFT.

Pads **pre-tokenized** samples (the output of ``alignment.hrm.data.format_hrm_sample``).
The on-the-fly / packed collator variants used by HRM-Text's pretraining
pipeline are out of scope for this SFT port.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase

IGNORE_LABEL_ID = -100


@dataclass
class DataCollatorForHRM:
    """Collate pre-tokenized HRM samples into padded batches.

    Expected input feature keys (produced by ``format_hrm_sample``):
        input_ids        (List[int])
        labels           (List[int])
        token_type_ids   (List[int])
        attention_mask   (List[int])
        prefix_len       (int)       - only used when prompt_loss_weight > 0

    Output batch keys:
        input_ids        (LongTensor)  [B, T]
        labels           (LongTensor)  [B, T]  -- padded with IGNORE_LABEL_ID
        token_type_ids   (LongTensor)  [B, T]  -- padded with 0
        attention_mask   (LongTensor)  [B, T]  -- padded with 0
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    prompt_loss_weight: float = 0.0
    pad_to_multiple_of: Optional[int] = 8

    def __call__(
        self, features: List[Dict[str, Union[List[int], int]]]
    ) -> Dict[str, torch.Tensor]:
        # Determine the target sequence length for this batch
        batch_max = max(len(f["input_ids"]) for f in features)
        if self.max_length is not None:
            batch_max = min(batch_max, self.max_length)
        if (
            self.pad_to_multiple_of is not None
            and batch_max % self.pad_to_multiple_of != 0
        ):
            batch_max = (
                batch_max // self.pad_to_multiple_of + 1
            ) * self.pad_to_multiple_of

        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        token_type_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []

        for feat in features:
            seq_len = min(len(feat["input_ids"]), batch_max)
            pad_len = batch_max - seq_len

            input_ids = feat["input_ids"][:seq_len]
            labels = feat["labels"][:seq_len]
            tti = feat["token_type_ids"][:seq_len]
            attn = feat["attention_mask"][:seq_len]

            # Apply prompt_loss_weight: scale down (or zero out) instruction labels.
            # When prompt_loss_weight == 0, format_hrm_sample already stores -100
            # for the prompt portion, so no adjustment is needed here.
            if self.prompt_loss_weight > 0.0 and "prefix_len" in feat:
                # labels already contain real IDs for the prefix region; the
                # actual weighting is applied by the trainer's compute_loss,
                # not here. This branch is a documented no-op pass-through
                # (matches HRM-Text's own hf_alignment.collator).
                pass

            # Pad
            input_ids_batch.append(
                input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
            )
            labels_batch.append(labels + [IGNORE_LABEL_ID] * pad_len)
            token_type_ids_batch.append(tti + [0] * pad_len)
            attention_mask_batch.append(attn + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        }


@dataclass
class DataCollatorForHRMWeightedJSON:
    """Collate pre-tokenized HRM samples produced by
    ``alignment.hrm.data.format_hrm_weighted_json_sample``.

    Same padding as ``DataCollatorForHRM`` for the shared keys
    (input_ids/labels/token_type_ids/attention_mask), plus:
        loss_weights   (List[float])  - padded with 0.0
        decision_mask  (List[bool])   - padded with False
        task           (str)          - converted to an integer ``task_ids`` tensor
                                        via ``task_to_id`` (unknown tasks -> -1)
    """

    tokenizer: PreTrainedTokenizerBase
    task_to_id: Dict[str, int]
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8

    def __call__(
        self, features: List[Dict[str, Union[List[int], List[float], List[bool], str]]]
    ) -> Dict[str, torch.Tensor]:
        batch_max = max(len(f["input_ids"]) for f in features)
        if self.max_length is not None:
            batch_max = min(batch_max, self.max_length)
        if (
            self.pad_to_multiple_of is not None
            and batch_max % self.pad_to_multiple_of != 0
        ):
            batch_max = (
                batch_max // self.pad_to_multiple_of + 1
            ) * self.pad_to_multiple_of

        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        token_type_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        loss_weights_batch: List[List[float]] = []
        decision_mask_batch: List[List[bool]] = []

        for feat in features:
            n = len(feat["input_ids"])
            seq_len = min(n, batch_max)
            start = n - seq_len  # left-truncate (keep the tail), matching
            # format_hrm_weighted_json_sample's own left-truncation -- this
            # branch should only fire if max_length here disagrees with the
            # max_seq_length used at data-prep time.
            pad_len = batch_max - seq_len

            input_ids_batch.append(
                feat["input_ids"][start:] + [self.tokenizer.pad_token_id or 0] * pad_len
            )
            labels_batch.append(feat["labels"][start:] + [IGNORE_LABEL_ID] * pad_len)
            token_type_ids_batch.append(feat["token_type_ids"][start:] + [0] * pad_len)
            attention_mask_batch.append(feat["attention_mask"][start:] + [0] * pad_len)
            loss_weights_batch.append(feat["loss_weights"][start:] + [0.0] * pad_len)
            decision_mask_batch.append(
                feat["decision_mask"][start:] + [False] * pad_len
            )

        task_ids = torch.tensor(
            [self.task_to_id.get(f["task"], -1) for f in features], dtype=torch.long
        )

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights_batch, dtype=torch.float),
            "decision_mask": torch.tensor(decision_mask_batch, dtype=torch.bool),
            "task_ids": task_ids,
        }
