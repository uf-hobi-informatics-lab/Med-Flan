import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from pynvml import *


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


class TorchAnomalyDetectionCallBack(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        torch.autograd.set_detect_anomaly(True)


class GpuUtilPrintCallBack(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and (state.global_step + 1) % 100 == 0:
            print(
                "[", datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"), "]\t", logs
            )

        if (state.global_step + 1) % 500 == 0:
            print(get_gpu_utilization())

    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        print(get_gpu_utilization())

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        print(get_gpu_utilization())


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class DeepSpeedCleanupCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)

        engine = None
        if trainer is not None:
            engine = getattr(trainer, "deepspeed", None)
            if engine is None:
                engine = getattr(trainer, "model_wrapped", None)

        if engine is not None and hasattr(engine, "destroy"):
            try:
                engine.destroy()
            except Exception as e:
                print(f"[DeepSpeedCleanup] engine.destroy() failed: {e}")

        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


class DeepSpeedCheckpointCleanupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        output_dir = args.output_dir
        if not os.path.isdir(output_dir):
            return

        ds_ckpts = []
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path) and re.fullmatch(r"global_step\d+", name):
                ds_ckpts.append(path)

        if len(ds_ckpts) <= 1:
            return

        ds_ckpts.sort(key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()))

        for ckpt in ds_ckpts[:-1]:
            shutil.rmtree(ckpt, ignore_errors=True)


# class DeepSpeedCheckpointCleanupCallback(TrainerCallback):
#     """
#     Callback to remove old DeepSpeed checkpoints after saving a new one.
#     Keeps only the most recent checkpoint to save disk space.
#     """

#     def __init__(self, keep_last_n: int = 1):
#         """
#         Args:
#             keep_last_n: Number of most recent checkpoints to keep. Default is 1.
#         """
#         self.keep_last_n = keep_last_n

#     def on_save(self, args, state, control, **kwargs):
#         """
#         Called after a checkpoint is saved. Removes old DeepSpeed checkpoints.
#         """
#         # Only run on main process to avoid race conditions
#         if not state.is_world_process_zero:
#             return

#         output_dir = Path(args.output_dir)
#         if not output_dir.exists():
#             return

#         # Find all checkpoint directories
#         checkpoint_dirs = sorted(
#             [d for d in output_dir.glob("checkpoint-*") if d.is_dir()], key=lambda x: int(x.name.split("-")[-1])
#         )

#         if len(checkpoint_dirs) <= self.keep_last_n:
#             return

#         # Remove old checkpoints
#         checkpoints_to_remove = checkpoint_dirs[: -self.keep_last_n]

#         for checkpoint_dir in checkpoints_to_remove:
#             # Check for DeepSpeed-specific directories (global_step*)
#             ds_dirs = list(checkpoint_dir.glob("global_step*"))

#             if ds_dirs:
#                 print(f"[DeepSpeedCheckpointCleanup] Removing old DeepSpeed checkpoint: {checkpoint_dir}")
#                 try:
#                     # Remove DeepSpeed checkpoint subdirectories
#                     for ds_dir in ds_dirs:
#                         if ds_dir.is_dir():
#                             shutil.rmtree(ds_dir)
#                             print(f"  Removed DeepSpeed directory: {ds_dir.name}")

#                     # Optionally remove the entire checkpoint directory if it only contains DS checkpoints
#                     # Check if directory is now empty or only contains a few small files
#                     remaining_items = list(checkpoint_dir.iterdir())
#                     if not remaining_items or all(
#                         item.is_file() and item.stat().st_size < 1024 * 1024 for item in remaining_items  # < 1MB
#                     ):
#                         shutil.rmtree(checkpoint_dir)
#                         print(f"  Removed entire checkpoint directory: {checkpoint_dir.name}")

#                 except Exception as e:
#                     print(f"[DeepSpeedCheckpointCleanup] Failed to remove {checkpoint_dir}: {e}")


# class PrintLossCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         # Access and print the loss
#         loss = args[0]
#         print(f"Step {state.global_step}: Loss = {loss}")


def fix_lr_scheduler_kwargs_float(training_args):
    if getattr(training_args, "lr_scheduler_kwargs").get("min_lr", None):
        training_args.lr_scheduler_kwargs["min_lr"] = float(
            training_args.lr_scheduler_kwargs["min_lr"]
        )

    if getattr(training_args, "lr_scheduler_kwargs").get("min_lr_rate", None):
        training_args.lr_scheduler_kwargs["min_lr_rate"] = float(
            training_args.lr_scheduler_kwargs["min_lr_rate"]
        )


def check_length(sample, tokenizer, max_pls, max_cls):
    prompt_length = tokenizer(
        sample["text_prompt"],
        return_tensors="pt",
    )[
        "input_ids"
    ].size(dim=-1)

    chosen_length = tokenizer(
        sample["text_chosen"],
        return_tensors="pt",
    )[
        "input_ids"
    ].size(dim=-1)

    max_pls.append(prompt_length), max_cls.append(chosen_length)
