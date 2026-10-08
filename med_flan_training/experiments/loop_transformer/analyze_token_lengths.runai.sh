#!/bin/bash
# RunAI submission for the token-length analysis (scripts/analyze_token_lengths.py).
#
# CPU-only, single process (no torchrun/GPU needed -- it only loads tokenizers
# and applies chat templates, no model forward pass). Requests a single GPU
# slot purely so it can land on the same node pool as training
# jobs do; the script itself never touches CUDA.
#
# Usage:
#   bash experiments/loop_transformer/analyze_token_lengths.runai.sh

set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-default}"

IMAGE="${IMAGE:-${USER}/ouro_train:base}"

BASE="/project"
ROOT=${BASE}

DATASET_PATH="/project/med_flan/data/med_flan_v1_md"
JOB_NAME="analyze-token-lengths-v1-md-1"

echo "[INFO] Submitting job: ${JOB_NAME}"
echo "[INFO] Dataset: ${DATASET_PATH}"
echo "[INFO] Setting Run:ai project: ${PROJECT_NAME}"

runai project set ${PROJECT_NAME}

GPU_NUM=1

LOG_DIR=/project/med_flan/log/${JOB_NAME}
STDOUT_LOG_DIR=/project/med_flan/log/stdout/${JOB_NAME}

RUN_CMD="python ${ROOT}/scripts/analyze_token_lengths.py \
    --dataset-path ${DATASET_PATH} \
    --splits train dev \
    --qwen-path /project/models/Qwen3-8B \
    --ouro-path /project/models/Ouro-2.6B-Thinking \
    --hrm-path /project/models/HRM-Text-1B \
    --condition direct \
    --num-proc 8"

echo "[DEBUG] RUN_CMD=${RUN_CMD}"

REMOTE_CMD='
set -euo pipefail

LOG_FILE="${STDOUT_LOG_DIR}/pod.log"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

set -x

${RUN_CMD}
'

runai training submit "${JOB_NAME}" \
    --node-pools "${NODE_POOL:-default}" \
    --auto-deletion-time-after-completion 24h \
    -i "${IMAGE}" \
    --gpu-devices-request ${GPU_NUM} \
    --cpu-core-request 16 \
    --cpu-memory-request 64Gi \
    --cpu-memory-limit 128Gi \
    --environment USER=${USER} \
    --environment ROOT=${ROOT} \
    --environment TMPDIR=/project/cache/temp \
    --environment HF_DATASETS_CACHE=/project/cache/dataset \
    --environment HF_HOME=/project/cache/huggingface \
    --environment HF_HUB_OFFLINE=1 \
    --environment TRANSFORMERS_OFFLINE=1 \
    --environment HF_DATASETS_OFFLINE=1 \
    --environment RUN_CMD="${RUN_CMD}" \
    --environment STDOUT_LOG_DIR=${STDOUT_LOG_DIR} \
    --existing-pvc claimname=${PVC_PROJECT:-project},path=/project \
    --existing-pvc claimname=${PVC_MODELS:-models},path=/project/models \
    --working-dir /project \
    --priority medium-high \
    --backoff-limit 3 \
    --image-pull-policy Always \
    --command -- bash -lc "${REMOTE_CMD}"

echo "[INFO] Submitted: ${JOB_NAME}"
echo "[INFO] Monitor: runai list jobs | grep ${JOB_NAME}"
echo "[INFO] Logs: runai logs ${JOB_NAME}"
