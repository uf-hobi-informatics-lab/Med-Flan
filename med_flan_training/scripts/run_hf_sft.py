#!/usr/bin/env python
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
"""Supervised fine-tuning (SFT) entry point for HRM-Text.

Launch with torchrun + DeepSpeed::

    torchrun --nproc_per_node=8 \\
        scripts/run_hf_sft.py recipes/hrm_1b/sft.yaml \\
        --deepspeed=recipes/accelerate_configs/deepspeed_zs2.json

Or with accelerate::

    accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero2.yaml \\
        scripts/run_hf_sft.py recipes/hrm_1b/sft.yaml

Config YAML must define ModelArguments, DataArguments, and SFTConfig fields
from ``alignment.hrm.configs``. See recipes/hrm_1b/sft.yaml for a reference.

Uses a plain HF Trainer (not trl.SFTTrainer): samples are pre-tokenized by
``format_hrm_sample``, which produces the ``token_type_ids`` field HRM's
PrefixLM attention needs (1 = bidirectional prefix/prompt, 0 = causal
response) -- a mechanism trl.SFTTrainer's own tokenization path doesn't
produce.
"""

import logging
import sys
from pathlib import Path


p = Path(__file__).parent.parent / "src"
sys.path.append(p.as_posix())

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DeepSpeedCheckpointCleanupCallback,
    GpuUtilPrintCallBack,
    H4ArgumentParser,
    fix_lr_scheduler_kwargs_float,
    get_checkpoint,
)
from alignment.context_extension import get_context_extension_kwargs
from alignment.hrm import (
    DataArguments,
    DataCollatorForHRM,
    DataCollatorForHRMWeightedJSON,
    HRMSFTTrainer,
    HRMWeightedJSONTrainer,
    ModelArguments,
    SFTConfig,
    apply_fixed_bp_cycles,
    format_hrm_sample,
    format_hrm_weighted_json_sample,
    get_datasets,
    load_hrm_tokenizer,
)
from alignment.weighted_json_loss import report_fallback_rates

logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # Parse arguments from YAML (+ optional CLI overrides)
    # ------------------------------------------------------------------
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()
    fix_lr_scheduler_kwargs_float(training_args)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model  : {model_args}")
    logger.info(f"Data   : {data_args}")
    logger.info(f"Train  : {training_args}")

    set_seed(training_args.seed)

    assert not (
        training_args.use_weighted_json_loss and training_args.prompt_loss_weight != 0.0
    ), (
        "use_weighted_json_loss and prompt_loss_weight are mutually exclusive -- both "
        "restructure how loss is weighted across the response."
    )
    assert not training_args.use_weighted_json_loss or data_args.use_chat_template, (
        "use_weighted_json_loss requires use_chat_template=True (it needs a "
        "'messages' + 'task' shaped example)."
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_hrm_tokenizer(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading datasets ...")
    raw_datasets = get_datasets(
        data_args.dataset_path,
        data_args.eval_dataset_path,
        num_proc=data_args.preprocessing_num_workers,
        dataset_splits=data_args.dataset_splits,
    )

    if training_args.use_weighted_json_loss:
        logger.info("Formatting dataset with HRM weighted-JSON prompt template ...")
        formatted_datasets = raw_datasets.map(
            lambda ex: format_hrm_weighted_json_sample(
                ex,
                tokenizer=tokenizer,
                condition=model_args.condition,
                max_seq_length=data_args.max_seq_length,
                scaffold_weight=training_args.weighted_json_scaffold_weight,
            ),
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Formatting (HRM weighted-JSON template)",
        )
        report_fallback_rates(formatted_datasets, threshold=0.01)
    else:
        logger.info("Formatting dataset with HRM prompt template ...")
        formatted_datasets = raw_datasets.map(
            lambda ex: format_hrm_sample(
                ex,
                tokenizer=tokenizer,
                condition=model_args.condition,
                max_seq_length=data_args.max_seq_length,
                prompt_loss_weight=training_args.prompt_loss_weight,
                use_chat_template=data_args.use_chat_template,
            ),
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Formatting (HRM template)",
        )

    train_dataset = formatted_datasets["train"]
    eval_dataset = formatted_datasets.get("test", None)

    logger.info(
        f"Train samples: {len(train_dataset)}"
        + (
            f"  Eval samples: {len(eval_dataset)}"
            if eval_dataset
            else "  (no eval split)"
        )
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Loading model ...")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ("auto", None)
        else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    # HRM-Text uses RoPE (no learned/absolute positional embedding matrix), so
    # max_position_embeddings is not baked into trained weights -- extending it is a
    # safe config override, not a shape mismatch against the checkpoint's weights.
    # use_cache is passed as a plain kwarg (not set on an explicit config object) so
    # it goes through from_pretrained's normal kwarg-absorption into the config,
    # same as the context-extension kwargs below.
    model_kwargs["use_cache"] = not training_args.gradient_checkpointing

    # Context-window extension: no-op if the checkpoint's own trained max length
    # already covers max_seq_length, else extends it (RoPE scaling via
    # config.rope_parameters, HrmTextRotaryEmbedding's dispatch key) -- see
    # alignment/context_extension.py.
    model_kwargs.update(
        get_context_extension_kwargs(
            model_args.model_name_or_path,
            data_args.max_seq_length,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    if training_args.gradient_checkpointing:
        # Non-reentrant checkpointing (PyTorch's default since ~2.4) strictly compares
        # recomputed activation tensors against the ones saved during the original
        # forward. At HRM-Text's long sequence lengths, SDPA's kernel dispatcher can
        # pick a different backend (math vs. flash/mem-efficient) between the forward
        # and the backward-time recompute depending on memory pressure, which trips
        # that check and raises `CheckpointError: Recomputed values ... have different
        # metadata`. use_reentrant=True uses the old-style checkpointing implementation,
        # which recomputes inside a custom autograd Function instead of comparing saved
        # vs. recomputed tensor metadata, so this mismatch can't occur.
        gc_kwargs = training_args.gradient_checkpointing_kwargs or {
            "use_reentrant": True
        }
        logger.info(f"Enabling gradient checkpointing with kwargs: {gc_kwargs}")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)

    # Fixed truncated-backprop window for fine-tuning: set once, held constant
    # for the whole run (no per-step warmup, unlike HRM-Text's pretraining
    # pipeline). No-ops on model architectures without this mechanism.
    apply_fixed_bp_cycles(model, training_args.bp_steps)

    # ------------------------------------------------------------------
    # Data collator
    # ------------------------------------------------------------------
    task_to_id = None
    if training_args.use_weighted_json_loss:
        task_to_id = {t: i for i, t in enumerate(sorted(set(train_dataset["task"])))}
        collator = DataCollatorForHRMWeightedJSON(
            tokenizer=tokenizer,
            task_to_id=task_to_id,
            max_length=data_args.max_seq_length,
            pad_to_multiple_of=8,
        )
    else:
        collator = DataCollatorForHRM(
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
            prompt_loss_weight=training_args.prompt_loss_weight,
            pad_to_multiple_of=8,
        )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer_cls = (
        HRMWeightedJSONTrainer
        if training_args.use_weighted_json_loss
        else HRMSFTTrainer
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if training_args.use_weighted_json_loss:
        trainer_kwargs["task_to_id"] = task_to_id
    trainer = trainer_cls(**trainer_kwargs)
    # trainer.add_callback(GpuUtilPrintCallBack())
    # trainer.add_callback(DeepSpeedCheckpointCleanupCallback())

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("*** Training ***")
    last_checkpoint = get_checkpoint(training_args)
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    if checkpoint:
        logger.info(f"Resuming from checkpoint: {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info("*** Saving model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Done ***")


if __name__ == "__main__":
    main()
