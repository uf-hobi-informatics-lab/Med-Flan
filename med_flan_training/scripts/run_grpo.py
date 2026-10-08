#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The Eli Lilly AADS Team. All rights reserved.

import logging
import sys
import traceback
from pathlib import Path


p = Path(__file__).parent.parent / "src"
sys.path.append(p.as_posix())

import torch
import transformers
from transformers import set_seed

from alignment import (
    DataArguments,
    DeepSpeedCheckpointCleanupCallback,
    GRPOConfig,
    H4ArgumentParser,
    ModelArguments,
    convert_to_grpo,
    fix_lr_scheduler_kwargs_float,
    get_checkpoint,
    get_datasets,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    reward_function_dict,
    sync_special_token_ids,
)
from trl import GRPOTrainer


logger = logging.getLogger(__name__)


def main():
    try:
        parser = H4ArgumentParser((ModelArguments, DataArguments, GRPOConfig))
        model_args, data_args, training_args = parser.parse()
        fix_lr_scheduler_kwargs_float(training_args)

        #######
        # Setup
        #######
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

        # Log on each process the small summary:
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Data parameters {data_args}")
        logger.info(f"Training/evaluation parameters {training_args}")

        # Set seed for reproducibility
        set_seed(training_args.seed)

        #####################################
        # Load tokenizer and process datasets
        #####################################
        data_args.truncation_side = (
            "left"  # Truncate from left to ensure we don't lose labels in final turn
        )
        tokenizer = get_tokenizer(model_args, data_args, training_args)

        ###############
        # Load datasets
        ###############
        grpo_dataset = get_datasets(
            data_args,
            splits=data_args.dataset_splits,
            configs=data_args.dataset_configs,
            columns_to_keep=[
                "messages",
                "chosen",
                "rejected",
                "prompt",
                "completion",
                "label",
                "task",  # carried through convert_to_grpo for task-routed rewards
            ],
        )

        if training_args.do_sft_to_grpo_conversion:
            grpo_dataset = grpo_dataset.map(
                convert_to_grpo,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=list(grpo_dataset["train"].features),
                desc="convert sft to grpo",
            )

        train_dataset = grpo_dataset["train"]
        eval_dataset = grpo_dataset["test"]

        #####################
        # prepare model args / PEFT config
        #####################
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        quantization_config = (
            get_quantization_config(model_args) if model_args.use_peft else None
        )

        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=training_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            quantization_config=quantization_config,
        )

        training_args.model_init_kwargs = model_kwargs
        peft_config = get_peft_config(model_args)

        ############### load reward functions ############################
        reward_function_list = []

        for func_name in training_args.reward_functions:
            logger.info(f"use reword function: {func_name}")
            if func_name == "push_pull_reward_function":
                _reward_function = reward_function_dict.get(func_name)(
                    n=training_args.num_generations
                )
                reward_function_list.append(_reward_function)
            elif func_name == "semantic_similarity_reward":
                # currently hardcoded model path -> the model is balance between performance and cost
                _reward_function = reward_function_dict.get(func_name)(
                    model_name="/project/models/Qwen3-Embedding-4B",
                    batch_size=8,
                    max_model_len=training_args.max_completion_length,
                    temp_dir="/cache/grpo_semantic_similarity_reward",
                )
                reward_function_list.append(_reward_function)
            else:
                reward_function_list.append(reward_function_dict.get(func_name, None))

        reward_function_list = [e for e in reward_function_list if e is not None]
        ###################################################################

        #########################
        # Instantiate GRPO trainer
        #########################
        # Multi-temperature rollout (MDP-GRPO): opt-in via the recipe's
        # temperature_schedule field. When set, MedGRPOTrainer overrides the vLLM
        # colocate generation path to sample each prompt's num_generations completions at
        # varied temperatures (raises within-group reward dispersion -> fewer zero-advantage
        # groups on deterministic 0/1 rewards). When unset, behaves identically to GRPOTrainer.
        if getattr(training_args, "temperature_schedule", None):
            from alignment.med_grpo_trainer import MedGRPOTrainer

            logger.info(
                f"Using MedGRPOTrainer with multi-temperature schedule: {training_args.temperature_schedule}"
            )
            grpo_trainer_cls = MedGRPOTrainer
        else:
            grpo_trainer_cls = GRPOTrainer

        grpo_trainer_kwargs = dict(
            model=model_args.model_name_or_path,
            reward_funcs=reward_function_list,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        trainer = grpo_trainer_cls(**grpo_trainer_kwargs)

        if data_args.special_tokens:
            sync_special_token_ids(trainer.model, tokenizer)

        # Add callback to remove old DeepSpeed checkpoints
        # By default, keeps only the last checkpoint. Adjust keep_last_n to keep more.
        # trainer.add_callback(DeepSpeedCheckpointCleanupCallback())

        ###############
        # Training loop
        ###############
        # Check for last checkpoint
        last_checkpoint = get_checkpoint(training_args)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(grpo_dataset["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("*** Training complete ***")

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        # kwargs = {
        #     # "finetuned_from": model_args.model_name_or_path, # modify for sfttrainer latest update
        #     "model_name": model_args.model_name_or_path,
        #     "dataset_name": "\n".join(list(data_args.dataset_mixer.keys())),
        #     "tags": ["alignment-handbook"],
        # }
        if trainer.accelerator.is_main_process:
            # trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference  `   `
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

        ##########
        # Evaluate
        ##########
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(grpo_dataset["test"])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        logger.info("*** Training complete! ***")
    except Exception as e:
        traceback.print_exc()
    finally:
        cleanup()
        sys.exit(1)


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
