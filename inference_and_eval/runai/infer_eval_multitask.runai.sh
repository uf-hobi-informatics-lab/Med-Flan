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
#
# BATCH_SIZE env var (unset by default -- run_generation.py's own default,
# everyone in one llm.chat() call) is forwarded the same way, e.g. to isolate
# a stuck/pathological example to its own call instead of one shared call
# for the whole dataset:
#   BATCH_SIZE=1 bash inference_and_eval/runai/infer_eval_multitask.runai.sh <model_path>
#
# BACKEND_ENGINE=hf (default: vllm) switches GEN_CMD to
# inference_and_eval.inference.run_generation_hf -- a plain-transformers
# generation path with no vLLM/xgrammar dependency, built for HRM
# (trust_remote_code custom architecture, PrefixLM token_type_ids handling
# vLLM's generic serving path can't apply -- see that module's docstring).
# CONDITION (the HRM condition value this checkpoint was trained with,
# e.g. "direct"/"cot"/"noisy"/"synth", or a composite like "synth,cot" --
# see hf_structured_infer.py's --condition) defaults to "direct" (the
# value every HRM checkpoint in this repo's sweeps is trained with);
# override via the CONDITION env var for a non-direct checkpoint. Only read
# when BACKEND_ENGINE=hf. Opt-in, not auto-selected for the hrm family: A/B
# against the vLLM path first, e.g. with a small --limit/--datasets run,
# before relying on it for a full sweep:
#   BACKEND_ENGINE=hf bash inference_and_eval/runai/infer_eval_multitask.runai.sh <hrm_model_path>
#
# DTYPE (default: empty) is forwarded to both backends' --dtype. Empty lets
# each backend use its own default (vLLM auto-derives from config.json
# torch_dtype; the HF path defaults to bfloat16). Pass bf16/bfloat16/fp16/
# fp32 to force it for either backend -- the vLLM side normalizes bf16->
# bfloat16 since vLLM rejects the shorthand. e.g.:
#   DTYPE=bf16 BACKEND_ENGINE=hf bash inference_and_eval/runai/infer_eval_multitask.runai.sh <hrm_model_path>
#
# BACKEND_ENGINE=native switches GEN_CMD to
# inference_and_eval.inference.run_generation_native -- a plain-transformers
# sweep that runs the model's own trust_remote_code generate() via
# native_infer.run(). Used for Ouro checkpoints with adaptive early-exit
# (the *_gated_* variant folders): vLLM's OuroForCausalLM loads the
# early_exit_gate weights but never calls them, always running the full fixed
# total_ut_steps loop and silently dropping the gate; the native path
# respects it. Auto-selected when VARIANT_DIR contains "_gated_" and the
# caller didn't explicitly set BACKEND_ENGINE -- so a gated checkpoint is
# routed automatically. Set explicitly to force it, or to A/B a gated
# checkpoint against vLLM (BACKEND_ENGINE=vllm) to see gated vs. forced-loop
# behavior. No grammar-constrained JSON / retry / repetition handling (the
# native path is the model's raw output); BATCH_SIZE is ignored on this path
# (native_infer processes one example at a time). e.g.:
#   BACKEND_ENGINE=native bash inference_and_eval/runai/infer_eval_multitask.runai.sh <ouro_gated_model_path>
#   # or just submit a *_gated_* path with no BACKEND_ENGINE set -- auto-detected.

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
BATCH_SIZE="${BATCH_SIZE:-}"
# Capture whether BACKEND_ENGINE was explicitly set by the caller BEFORE this
# script's own default below -- so the _gated_ auto-detection (after VARIANT_DIR
# is known) only kicks in when the caller didn't already choose a backend.
_BACKEND_ENGINE_SET=${BACKEND_ENGINE:+yes}
BACKEND_ENGINE="${BACKEND_ENGINE:-vllm}"
CONDITION="${CONDITION:-direct}"
DTYPE="${DTYPE:-}"

# vLLM concurrency/memory knobs to counter preemption thrash on HRM-Text
# (B300, single-GPU). Empty by default -> vLLM's own defaults
# (max_num_seqs=256, gpu_memory_utilization=0.9). For HRM the block after
# FAMILY detection sets conservative defaults; override per-run, e.g.
# MAX_NUM_SEQS=16 bash ...  Only applied to the vLLM branch (the HF path has
# its own memory-budgeted sub-chunking -- hf_structured_infer._safe_max_new_tokens).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"

case "${BACKEND_ENGINE}" in
  vllm|hf|native) ;;
  *)
    echo "[ERROR] unknown BACKEND_ENGINE '${BACKEND_ENGINE}' -- expected 'vllm', 'hf', or 'native'" >&2
    exit 1
    ;;
esac

PROJECT_NAME="${PROJECT_NAME:-default}"
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

# Auto-detect a gated Ouro checkpoint (variant dir name contains "_gated_")
# and route it to the native (dynamic-round) backend -- vLLM's OuroForCausalLM
# loads the early_exit_gate weights but never calls them in forward(), so it
# always runs the full fixed total_ut_steps loop, silently dropping the
# adaptive-early-exit feature the checkpoint was trained to use. The native
# path (run_generation_native -> native_infer.run) runs the model's own
# trust_remote_code generate(), which DOES respect the gate. Only auto-selects
# when the caller didn't explicitly set BACKEND_ENGINE (an explicit vllm/hf
# still wins, e.g. to A/B a gated checkpoint's gated vs. forced-full-loop
# behavior). See native_infer.py's module docstring for the vLLM limitation.
if [ -z "${_BACKEND_ENGINE_SET}" ] && echo "${VARIANT_DIR}" | grep -q "_gated_"; then
  echo "[INFO] ${VARIANT_DIR}: _gated_ detected -> BACKEND_ENGINE=native (Ouro adaptive early-exit; vLLM ignores the gate)"
  BACKEND_ENGINE=native
fi

case "${FAMILY}" in
  hrm)
    IMAGE="${IMAGE:-${USER}/hrm_train:base}"
    ;;
  ouro|qwen)
    IMAGE="${IMAGE:-${USER}/ouro_train:base}"
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

# MAIN_FOLDER containing an "md" token (v1_md, v1_md_upsampled, ...) means
# this checkpoint was trained on messages_md (markdown-rendered) data --
# eval must use the matching "<dataset>_md" huggingface_format/ variant
# (see run_generation.py's --markdown) so train/eval prompt formatting
# doesn't silently mismatch. Token-matched (not a substring grep) for the
# same reason sweep_eval.runai.sh's stage1/stage2 check is -- avoids a
# false positive on e.g. a hypothetical "v1_medium" folder.
USE_MARKDOWN=false
IFS='_' read -ra _MF_TOKENS <<< "${MAIN_FOLDER}"
for T in "${_MF_TOKENS[@]}"; do
  [ "${T}" = "md" ] && USE_MARKDOWN=true && break
done

MODEL_TAG="${MAIN_FOLDER}_${VARIANT_DIR}_${CKPT_DIR}"
if [ -n "${RANK}" ]; then
  MODEL_TAG="${MODEL_TAG}_rank$(printf '%02d' "${RANK}")"
fi
# non-default temperature gets its own results dir, so a sampling run never
# overwrites the greedy (temperature=0.0) baseline for the same checkpoint.
if [ "${TEMPERATURE}" != "0.0" ]; then
  MODEL_TAG="${MODEL_TAG}_temp${TEMPERATURE}"
fi
# The filename tag run_generation.py's model_tag() actually writes the
# per-dataset completion JSONs under -- <variant>_<checkpoint>, WITHOUT the
# MAIN_FOLDER prefix or the _rankNN/_temp suffixes MODEL_TAG carries above
# (MODEL_TAG is for the results DIR + eval CSV name; the JSON filenames come
# from model_tag(MODEL_PATH) in run_generation.py). Passed to run_eval_csv as
# --model-tag so its glob matches the actual <FILE_TAG>__<dataset>.json files
# instead of falling back to the bare <dataset>.json pattern (which never
# matches and silently skips every dataset in the eval step).
FILE_TAG="${VARIANT_DIR}_${CKPT_DIR}"
# RunAI rejects workload names over 50 chars -- a fully descriptive name
# (main_folder + variant + checkpoint + rank + temp) can exceed that for
# longer variant names, so truncate and append a short deterministic
# checksum suffix (derived from the FULL untruncated name) rather than just
# chopping the end blindly -- two different combos that happen to share a
# long common prefix would otherwise silently collide on the same job name
# once both are cut to the same length. OUT_DIR (below) stays keyed off the
# untruncated MODEL_TAG, so results/ directory and file names remain fully
# descriptive regardless of this.
FULL_JOB_NAME=$(echo "${MODEL_TAG}-infer-eval" | tr '[:upper:]_.' '[:lower:]--' | tr -s '-' | sed -E 's/^-+|-+$//g')
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
echo "[INFO] Batch size: ${BATCH_SIZE:-<default: everyone in one call>}"
echo "[INFO] Eval dataset variant: $([ "${USE_MARKDOWN}" = true ] && echo markdown || echo plain)"
echo "[INFO] Backend engine: ${BACKEND_ENGINE}$([ "${BACKEND_ENGINE}" = hf ] && echo " (condition=${CONDITION})")$([ "${BACKEND_ENGINE}" = native ] && echo " (trust_remote_code, Ouro adaptive early-exit)")"
echo "[INFO] Dtype: ${DTYPE:-<default: vllm auto-derive / hf bfloat16>}"
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

if [ "${BACKEND_ENGINE}" = hf ]; then
  GEN_CMD="python3 -m inference_and_eval.inference.run_generation_hf --model ${MODEL_PATH} --condition ${CONDITION} --temperature ${TEMPERATURE} --top-p ${TOP_P} --out-dir ${OUT_DIR}"
elif [ "${BACKEND_ENGINE}" = native ]; then
  # Native (trust_remote_code) path -- runs Ouro's own generate() so the
  # adaptive early-exit gate actually fires (vLLM ignores it). See
  # run_generation_native.py / native_infer.py. --trust-remote-code is on by
  # default in that wrapper (Ouro ships custom modeling code); --condition is
  # HRM-only and not passed here.
  GEN_CMD="python3 -m inference_and_eval.inference.run_generation_native --model ${MODEL_PATH} --trust-remote-code --temperature ${TEMPERATURE} --top-p ${TOP_P} --out-dir ${OUT_DIR}"
else
  GEN_CMD="python3 -m inference_and_eval.inference.run_generation --model ${MODEL_PATH} --backend ${BACKEND} --temperature ${TEMPERATURE} --top-p ${TOP_P} --out-dir ${OUT_DIR}"
fi
if [ "${USE_MARKDOWN}" = true ]; then
  GEN_CMD="${GEN_CMD} --markdown"
fi
if [ -n "${BATCH_SIZE}" ]; then
  # --batch-size is a vLLM/HF knob (caps examples per llm.chat()/generate()
  # call). The native path (native_infer.run) processes one example at a time
  # by design -- no batching to cap -- so silently ignore BATCH_SIZE there
  # instead of passing a flag the wrapper doesn't accept.
  if [ "${BACKEND_ENGINE}" != native ]; then
    GEN_CMD="${GEN_CMD} --batch-size ${BATCH_SIZE}"
  fi
fi
# DTYPE forwarded to both backends' --dtype: vLLM accepts bfloat16/float16/
# float32/auto (and the run_generation.py side normalizes bf16->bfloat16);
# the HF side accepts bf16/bfloat16/fp16/fp32. Empty (default) lets each
# backend use its own default (vLLM auto-derives from config.json; HF
# defaults to bfloat16) -- same behavior as before this passthrough.
if [ -n "${DTYPE}" ]; then
  GEN_CMD="${GEN_CMD} --dtype ${DTYPE}"
fi

EVAL_CMD="python3 -m inference_and_eval.inference.run_eval_csv --completions-dir ${OUT_DIR} --model-tag ${FILE_TAG} --out ${OUT_DIR}/${MODEL_TAG}_eval_results.csv"

echo "[DEBUG] GEN_CMD=${GEN_CMD}"
echo "[DEBUG] EVAL_CMD=${EVAL_CMD}"

# Mirror stdout/stderr to a file on the shared PVC (in addition to the terminal /
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
    --node-pools "${NODE_POOL:-default}" \
    --auto-deletion-time-after-completion 1h \
    -i "${IMAGE}" \
    --gpu-devices-request ${GPU_NUM} \
    --cpu-core-request 16 \
    --cpu-memory-request 200Gi \
    --cpu-memory-limit 400Gi \
    --environment USER=${USER} \
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
    --existing-pvc claimname=${PVC_PROJECT:-project},path=/projects \
    --existing-pvc claimname=${PVC_QWEN:-qwen},path=/data/qwen \
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
