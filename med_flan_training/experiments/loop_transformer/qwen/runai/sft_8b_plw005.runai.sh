#!/bin/bash
# RunAI submission for Qwen3-8B SFT (sft_8b_plw005_v1_md) -- scripts/run_sft.py.
#
# Single node, 4 GPUs (B300). FSDP config comes from the recipe yaml itself
# (fsdp/fsdp_config fields), so no --fsdp CLI flags are needed here.
#
# Usage:
#   bash experiments/loop_transformer/qwen/runai/sft_8b_plw005.runai.sh

set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-default}"

IMAGE="${IMAGE:-${USER}/ouro_train:base}"

BASE="/project"
ROOT=${BASE}

TRAIN_CONF=${ROOT}/recipes/loop_transformer/qwen/sft_8b_plw005.yaml
TRAIN_SCRIPT=${ROOT}/scripts/run_sft.py
JOB_NAME="qwen-sft-8b-plw005-v1-md-1"

echo "[INFO] Submitting job: ${JOB_NAME}"
echo "[INFO] Train config: ${TRAIN_CONF}"
echo "[INFO] Setting Run:ai project: ${PROJECT_NAME}"

runai project set ${PROJECT_NAME}

# Single node only -- B300 nodes have 4 GPUs each.
GPU_NUM=4

LOG_DIR=/project/med_flan/log/${JOB_NAME}
STDOUT_LOG_DIR=/project/med_flan/log/stdout/${JOB_NAME}

RUN_CMD="torchrun --nproc_per_node=${GPU_NUM} --nnodes=1 ${TRAIN_SCRIPT} ${TRAIN_CONF} --tee=3"

echo "[DEBUG] RUN_CMD=${RUN_CMD}"

# Mirror stdout/stderr to a file on the weka PVC (in addition to the terminal /
# RunAI-captured pod log) -- survives pod crashes/restarts and stays readable
# after the job/pod is gone.
REMOTE_CMD='
set -euo pipefail

LOG_FILE="${STDOUT_LOG_DIR}/pod.log"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

set -x

echo "[INFO] Using: MASTER=${MASTER_ADDR:-127.0.0.1}:${MASTER_PORT:-23456}";

source ${ROOT}/experiments/wandb.sh

${RUN_CMD}
'

runai training submit "${JOB_NAME}" \
    --node-pools "${NODE_POOL:-default}" \
    --auto-deletion-time-after-completion 24h \
    -i "${IMAGE}" \
    --gpu-devices-request ${GPU_NUM} \
    --cpu-core-request 32 \
    --cpu-memory-request 500Gi \
    --cpu-memory-limit 1000Gi \
    --environment USER=${USER} \
    --environment ROOT=${ROOT} \
    --environment TMPDIR=/project/cache/temp \
    --environment HF_DATASETS_CACHE=/project/cache/dataset \
    --environment HF_HOME=/project/cache/huggingface \
    --environment TRITON_HOME=/project/cache/triton \
    --environment TRITON_CACHE_DIR=/project/cache/triton/cache \
    --environment TRITON_DUMP_DIR=/project/cache/triton/dump \
    --environment TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --environment BITSANDBYTES_SKIP_LOAD=1 \
    --environment BNB_CUDA_VERSION=130 \
    --environment BITSANDBYTES_FORCE_CUDA_VERSION=130 \
    --environment OMP_NUM_THREADS=1 \
    --environment HF_HUB_OFFLINE=1 \
    --environment TRANSFORMERS_OFFLINE=1 \
    --environment HF_DATASETS_OFFLINE=1 \
    --environment WANDB_PROJECT=loop_transformer_v1_md \
    --environment WANDB_NAME=${JOB_NAME} \
    --environment RUN_CMD="${RUN_CMD}" \
    --environment STDOUT_LOG_DIR=${STDOUT_LOG_DIR} \
    --existing-pvc claimname=${PVC_PROJECT:-project},path=/project \
    --existing-pvc claimname=${PVC_MODELS:-models},path=/project/models \
    --working-dir /project \
    --priority medium-high \
    --backoff-limit 3 \
    --image-pull-policy Always \
    --large-shm \
    --command -- bash -lc "${REMOTE_CMD}"

echo "[INFO] Submitted: ${JOB_NAME}"
echo "[INFO] Monitor: runai list jobs | grep ${JOB_NAME}"
echo "[INFO] Logs: runai logs ${JOB_NAME}"
