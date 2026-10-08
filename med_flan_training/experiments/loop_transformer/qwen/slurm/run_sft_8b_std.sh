#!/usr/bin/env bash
# Runs the Qwen3-8B SFT (std) experiment inside the container launched by
# sft_8b_std.slurm.
set -euo pipefail

whoami
pwd
ds_report || true

HOME=/project
ROOT=${HOME}
SCRIPTPATH=${ROOT}/experiments
source ${SCRIPTPATH}/wandb.sh

export TMPDIR="/cache"
export TRITON_HOME=${TMPDIR}/triton
export TRITON_CACHE_DIR=${HOME}/cache/triton/cache
export TRITON_DUMP_DIR=${HOME}/cache/triton/dump
export HF_DATASETS_CACHE=${HOME}/cache/dataset
export HF_HOME=${HOME}/cache/huggingface
# 32768-token sequences at batch=1 leave the allocator with large, oddly-sized
# blocks; expandable_segments avoids fragmentation-driven OOMs where a lot of
# memory is "reserved but unallocated" (as opposed to genuinely out of room).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_CONF="${ROOT}/recipes/loop_transformer/qwen/sft_8b_std.yaml"
echo "TRAIN_CONF: $TRAIN_CONF"

export WANDB_PROJECT="loop_transformer_qwen"
export WANDB_NAME="sft_8b_std"

echo "running with SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-}"
echo "running with SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-}"

torchrun \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    --nnodes="${SLURM_JOB_NUM_NODES:-1}" \
    --node_rank="${SLURM_NODEID:-0}" \
    --master_addr="$PRIMARY" \
    --master_port="$PRIMARY_PORT" \
    ${ROOT}/scripts/run_sft.py \
    "$TRAIN_CONF" \
    --tee=3
