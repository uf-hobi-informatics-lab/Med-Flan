#!/bin/bash
# RunAI submission: vLLM inference + evaluation over the multitask eval suite,
# in one job -- inference_and_eval.inference.run_generation (per-dataset
# completions, see MAIN_DATASET_SPECS/OOD_DATASET_SPECS there) followed by
# inference_and_eval.inference.run_eval_csv (each task's real scorer,
# long-format metrics CSV). See inference_and_eval/inference/README.md for
# what each step does.
#
# IMAGE NOTE: vLLM has to actually support the checkpoint's architecture, so
# this reuses the SFT training images rather than the plain genai/nv_vllm
# image. Picked automatically from MODEL_PATH's variant directory name
# (the "ouro_"/"hrm_"/"qwen_" prefix in e.g. ".../ouro_1_4b_std_plw005/
# checkpoint-25860"):
#   - ouro_train:base  for Ouro and Qwen checkpoints
#   - hrm_train:base   for HRM checkpoints
#
# Runs every dataset in DATASET_SPECS (inference_and_eval/inference/run_generation.py)
# -- no subsetting, this is meant as the full eval sweep for one checkpoint.
#
# Single node, 1 GPU (B300) -- vLLM inference for a <=8B checkpoint doesn't
# need tensor-parallel sharding. Bump GPU_NUM and add
# --tensor-parallel-size to GEN_CMD below if a checkpoint needs more.
#
# Usage:
#   bash inference_and_eval/runai/infer_eval_multitask.runai.sh \
#       /projects/med_flan/models/v1_md/ouro_1_4b_std_plw005_1/checkpoint-25860 [rank]
#
# Optional [rank] is purely a label (e.g. this checkpoint's 1-indexed rank
# among a variant's checkpoints sorted by step) folded into MODEL_TAG/OUT_DIR
# so results stay distinguishable when driven by sweep_eval.runai.sh.
#
# TEMPERATURE/TOP_P env vars (default 0.0/1.0 -- greedy, matching
# run_generation.py's own default) are passed straight through to GEN_CMD,
# so a future non-greedy sampling run doesn't need this file edited:
#   TEMPERATURE=0.5 TOP_P=0.95 bash inference_and_eval/runai/infer_eval_multitask.runai.sh <model_path>
# Folded into MODEL_TAG/OUT_DIR whenever TEMPERATURE isn't the 0.0 default,
# so a sampling run never overwrites the greedy baseline's results.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_path> [rank]" >&2
  echo "  e.g. $0 /projects/med_flan/models/v1_md/ouro_1_4b_std_plw005_1/checkpoint-25860" >&2
  exit 1
fi

MODEL_PATH="$1"
RANK="${2:-}"
BACKEND=xgrammar
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"

PROJECT_NAME="genai-gs"
BASE="/projects"
ROOT=${BASE}/med_flan
# repo root -- inference_and_eval/ (the module run via `python -m`) lives here,
# one level below ROOT, so this is what we `cd` into before invoking python.
REPO_ROOT=${ROOT}/meddata

# variant dir name (e.g. "ouro_1_4b_std_plw005") and checkpoint dir name
# (e.g. "checkpoint-25860") -- everything else (JOB_NAME, IMAGE, output dir)
# is derived from these two so nothing needs editing per checkpoint.
CKPT_DIR=$(basename "${MODEL_PATH}")
VARIANT_DIR=$(basename "$(dirname "${MODEL_PATH}")")
FAMILY=$(echo "${VARIANT_DIR}" | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]')

case "${FAMILY}" in
  hrm)
    IMAGE=runartifacts.am.lilly.com/lillypod/l069561/hrm_train:base
    ;;
  ouro|qwen)
    IMAGE=runartifacts.am.lilly.com/lillypod/l069561/ouro_train:base
    ;;
  *)
    echo "[ERROR] could not determine image family from MODEL_PATH's variant dir '${VARIANT_DIR}'" \
         "(expected it to start with ouro_/hrm_/qwen_)" >&2
    exit 1
    ;;
esac

# e.g. "v1_md_ouro_1_4b_std_plw005_checkpoint-25860_rank05" -- used for the
# eval_results.csv filename and (sanitized) the RunAI job name. MAIN_FOLDER
# (e.g. "v1_md") is folded in because variant dir names repeat across main
# folders (v1/v1_md/v1_md_upsampled) -- OUT_DIR below also nests results
# under MAIN_FOLDER as its own directory level, but MODEL_TAG keeps the
# prefix too so filenames stay unambiguous if ever copied out of that layout.
MAIN_FOLDER=$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")
MODEL_TAG="${MAIN_FOLDER}_${VARIANT_DIR}_${CKPT_DIR}"
if [ -n "${RANK}" ]; then
  MODEL_TAG="${MODEL_TAG}_rank$(printf '%02d' "${RANK}")"
fi
# non-default temperature gets its own results dir, so a sampling run never
# overwrites the greedy (temperature=0.0) baseline for the same checkpoint.
if [ "${TEMPERATURE}" != "0.0" ]; then
  MODEL_TAG="${MODEL_TAG}_temp${TEMPERATURE}"
fi
# RunAI rejects workload names over 50 chars -- a fully descriptive name
# (main_folder + variant + checkpoint + rank + temp) can exceed that for
# longer variant names, so truncate and append a short deterministic
# checksum suffix (derived from the FULL untruncated name) rather than just
# chopping the end blindly -- two different combos that happen to share a
# long common prefix would otherwise silently collide on the same job name
# once both are cut to the same length. OUT_DIR (below) stays keyed off the
# untruncated MODEL_TAG, so results/ directory and file names remain fully
# descriptive regardless of this.
FULL_JOB_NAME=$(echo "${MODEL_TAG}-infer-eval" | tr '[:upper:]_' '[:lower:]-' | tr -s '-' | sed -E 's/^-+|-+$//g')
MAX_JOB_NAME_LEN=50
if [ ${#FULL_JOB_NAME} -gt ${MAX_JOB_NAME_LEN} ]; then
  SUFFIX="-$(echo -n "${FULL_JOB_NAME}" | cksum | cut -d' ' -f1)"
  KEEP=$((MAX_JOB_NAME_LEN - ${#SUFFIX}))
  PREFIX="${FULL_JOB_NAME:0:${KEEP}}"
  PREFIX="${PREFIX%-}"
  JOB_NAME="${PREFIX}${SUFFIX}"
else
  JOB_NAME="${FULL_JOB_NAME}"
fi

echo "[INFO] Submitting job: ${JOB_NAME}"
echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Family: ${FAMILY} -> image ${IMAGE}"
echo "[INFO] Sampling: temperature=${TEMPERATURE} top_p=${TOP_P}"
echo "[INFO] Setting Run:ai project: ${PROJECT_NAME}"

runai project set ${PROJECT_NAME}

# Single node, single GPU -- see IMAGE NOTE above for when to raise this.
GPU_NUM=1

LOG_DIR=/projects/med_flan/log/${JOB_NAME}
STDOUT_LOG_DIR=/projects/med_flan/log/stdout/${JOB_NAME}
# Two-level results layout: main_folder (v1, v1_md, ...) / variant+checkpoint
# -- strip MODEL_TAG's own leading "${MAIN_FOLDER}_" so it isn't repeated in
# both the parent dir and the leaf dir name.
OUT_DIR=/projects/med_flan/results/${MAIN_FOLDER}/${MODEL_TAG#${MAIN_FOLDER}_}

GEN_CMD="python3 -m inference_and_eval.inference.run_generation --model ${MODEL_PATH} --backend ${BACKEND} --temperature ${TEMPERATURE} --top-p ${TOP_P} --out-dir ${OUT_DIR}"
EVAL_CMD="python3 -m inference_and_eval.inference.run_eval_csv --completions-dir ${OUT_DIR} --out ${OUT_DIR}/${MODEL_TAG}_eval_results.csv"

echo "[DEBUG] GEN_CMD=${GEN_CMD}"
echo "[DEBUG] EVAL_CMD=${EVAL_CMD}"

# Mirror stdout/stderr to a file on the weka PVC (in addition to the terminal /
# RunAI-captured pod log) -- survives pod crashes/restarts and stays readable
# after the job/pod is gone.
REMOTE_CMD='
set -euo pipefail

LOG_FILE="${STDOUT_LOG_DIR}/pod.log"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

set -x

cd ${REPO_ROOT}

echo "[STEP] generation"
${GEN_CMD}

echo "[STEP] evaluation"
${EVAL_CMD}
'

# If this job name already exists in the project, skip instead of failing.
set +e
SUBMIT_OUTPUT=$(runai training submit "${JOB_NAME}" \
    --node-pools weka-ready \
    --auto-deletion-time-after-completion 1h \
    -i "${IMAGE}" \
    --gpu-devices-request ${GPU_NUM} \
    --cpu-core-request 16 \
    --cpu-memory-request 200Gi \
    --cpu-memory-limit 400Gi \
    --environment USER=xi.yang5 \
    --environment ROOT=${ROOT} \
    --environment REPO_ROOT=${REPO_ROOT} \
    --environment TMPDIR=/projects/cache/temp \
    --environment HF_DATASETS_CACHE=/projects/cache/dataset \
    --environment HF_HOME=/projects/cache/huggingface \
    --environment TRITON_HOME=/projects/cache/triton \
    --environment TRITON_CACHE_DIR=/projects/cache/triton/cache \
    --environment TRITON_DUMP_DIR=/projects/cache/triton/dump \
    --environment OMP_NUM_THREADS=1 \
    --environment HF_HUB_OFFLINE=1 \
    --environment TRANSFORMERS_OFFLINE=1 \
    --environment HF_DATASETS_OFFLINE=1 \
    --environment GEN_CMD="${GEN_CMD}" \
    --environment EVAL_CMD="${EVAL_CMD}" \
    --environment STDOUT_LOG_DIR=${STDOUT_LOG_DIR} \
    --existing-pvc claimname=weka-genai-gs-proj,path=/projects \
    --existing-pvc claimname=weka-genai-gs-qwen,path=/data/qwen \
    --working-dir /projects \
    --priority medium-high \
    --backoff-limit 3 \
    --image-pull-policy Always \
    --large-shm \
    --command -- bash -lc "${REMOTE_CMD}" 2>&1)
SUBMIT_STATUS=$?
set -e
echo "${SUBMIT_OUTPUT}"

if [ ${SUBMIT_STATUS} -ne 0 ]; then
  if echo "${SUBMIT_OUTPUT}" | grep -q "already exists"; then
    echo "[INFO] Skipping: job '${JOB_NAME}' already exists in the project."
    exit 0
  fi
  echo "[ERROR] Failed to submit ${JOB_NAME}" >&2
  exit ${SUBMIT_STATUS}
fi

echo "[INFO] Submitted: ${JOB_NAME}"
echo "[INFO] Monitor: runai list jobs | grep ${JOB_NAME}"
echo "[INFO] Logs: runai logs ${JOB_NAME}"
echo "[INFO] Results: ${OUT_DIR}/${MODEL_TAG}_eval_results.csv (after the job completes)"
