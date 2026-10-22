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
"""Plain ``transformers.Trainer`` subclass for HRM-Text SFT that logs
``mean_token_accuracy``.

``run_hf_sft.py`` uses a vanilla ``Trainer`` (not ``trl.SFTTrainer``) because
HRM-Text's PrefixLM attention needs the pre-tokenized ``token_type_ids``
field ``format_hrm_sample``/``DataCollatorForHRM`` produce -- a shape
``SFTTrainer``'s own tokenization path doesn't support. That means it also
misses ``SFTTrainer``'s (and ``PLWTrainer``'s, see ``alignment.plw_trainer``)
built-in per-token accuracy logging. This reproduces that: buffer per-step
accuracy in ``self._metrics[mode]`` inside ``compute_loss``, then average and
flush into the log dict in ``log()`` -- the exact pattern ``trl.SFTTrainer``
uses for its own metrics.
"""

from collections import defaultdict

from torch.nn import CrossEntropyLoss
from transformers import Trainer

from .collator import IGNORE_LABEL_ID


class HRMSFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if model.training else "eval"
        # Saved before the base implementation runs -- it pops "labels" out of
        # `inputs` when a label_smoother/compute_loss_func is configured (neither
        # is, here, but this keeps the accuracy calc correct if that ever changes).
        labels = inputs["labels"]

        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:].to(shift_logits.device)
        mask = shift_labels != IGNORE_LABEL_ID

        predictions = shift_logits.argmax(dim=-1)
        correct_tokens = ((predictions == shift_labels) & mask).sum()
        total_tokens = mask.sum()

        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)

        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()


class HRMWeightedJSONTrainer(HRMSFTTrainer):
    """Structural analog of ``alignment.weighted_json_loss.WeightedJSONTrainer`` for
    HRM's plain-``Trainer`` pipeline: bypasses HF's built-in labels-based loss (the
    model is called without ``labels`` so it never computes its own unweighted CE),
    computes ``sum(loss_weights * CE) / sum(loss_weights)`` manually, and logs
    decision-token-only accuracy plus per-task accuracy -- mirroring that trainer's
    ``compute_loss`` exactly, just against HRM's pre-tokenized
    ``loss_weights``/``decision_mask``/``task_ids`` collator output instead of trl's.
    """

    def __init__(self, *args, task_to_id, **kwargs):
        self.task_to_id = task_to_id
        self.loss_fct = CrossEntropyLoss(reduction="none")
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        # Same rationale as WeightedJSONTrainer/PLWTrainer: remove_unused_columns
        # would otherwise strip these before the collator ever sees them.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "labels",
                "token_type_ids",
                "attention_mask",
                "loss_weights",
                "decision_mask",
                "task",
            ]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if model.training else "eval"

        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        decision_mask = inputs.pop("decision_mask")
        task_ids = inputs.pop("task_ids")

        outputs = model(**inputs, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        V = logits.size(-1)
        flat = logits[:, :-1, :].reshape(-1, V)
        shift_labels = labels[:, 1:].to(flat.device)
        shift_weights = loss_weights[:, 1:].to(flat.device, flat.dtype)
        shift_decision = decision_mask[:, 1:].to(flat.device)

        token_losses = self.loss_fct(flat, shift_labels.reshape(-1))
        denom = shift_weights.sum() + 1e-9
        loss = token_losses.float() @ shift_weights.reshape(-1).float() / denom

        self._metrics[mode]["loss_weighted"].append(loss.item())

        predictions = flat.argmax(dim=-1).view(logits.size(0), -1)
        mask = shift_decision
        correct = (predictions == shift_labels) & mask
        total_tokens = self.accelerator.gather_for_metrics(mask.sum())
        correct_tokens = self.accelerator.gather_for_metrics(correct.sum())
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["decision_token_accuracy"].append(accuracy)

        shift_task_ids = task_ids.unsqueeze(1).expand(-1, mask.size(1)).to(flat.device)
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
