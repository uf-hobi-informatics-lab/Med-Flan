#!/bin/bash
# Submit one infer+eval RunAI job (via infer_eval_multitask.runai.sh) per
# (model variant, selected checkpoint rank) under
# /projects/med_flan/models/<main_folder>/*/checkpoint-*.
#
# "Rank" = a checkpoint's 1-indexed position when a variant's checkpoints
# are sorted ascending by step number (rank 1 = earliest/smallest step).
# Which ranks to submit is set by RANKS below.
#
# Usage:
#   bash inference_and_eval/runai/sweep_eval.runai.sh <main_folder> [--families f1,f2,...] [--yes]
#     e.g. bash inference_and_eval/runai/sweep_eval.runai.sh v1_md
#          bash inference_and_eval/runai/sweep_eval.runai.sh v1_md --yes   # actually submit
#          bash inference_and_eval/runai/sweep_eval.runai.sh v1_md --families qwen,ouro
#          bash inference_and_eval/runai/sweep_eval.runai.sh v1_md --families hrm --yes
#
# --families restricts the sweep to variants whose family (the ouro_/hrm_/
# qwen_ prefix on the variant dir name, same as infer_eval_multitask.runai.sh's
# own FAMILY detection) is in the given comma-separated list. Omit it to sweep
# every family, as before.
#
# Dry run by default (prints what would be submitted, no runai calls) --
# pass --yes to actually submit. A single main_folder can contain ~20
# variants x 6 ranks below = 100+ individual GPU jobs, so this is meant to
# be reviewed before firing.
#
# TEMPERATURE/TOP_P env vars are forwarded to infer_eval_multitask.runai.sh
# (see that script's header) -- e.g. to sweep a non-greedy decode instead of
# the temperature=0.0 default:
#   TEMPERATURE=0.5 TOP_P=0.95 bash inference_and_eval/runai/sweep_eval.runai.sh v1_md --yes

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <main_folder> [--families f1,f2,...] [--yes]" >&2
  echo "  e.g. $0 v1_md   (v1, v1_md, v1_md_upsampled, ...)" >&2
  echo "  e.g. $0 v1_md --families qwen,ouro" >&2
  exit 1
fi

MAIN_FOLDER="$1"
shift

KNOWN_FAMILIES=(hrm ouro qwen)
FAMILIES=()
DO_SUBMIT=false
while [ $# -gt 0 ]; do
  case "$1" in
    --yes)
      DO_SUBMIT=true
      shift
      ;;
    --families)
      IFS=',' read -ra FAMILIES <<< "${2:-}"
      shift 2
      ;;
    --families=*)
      IFS=',' read -ra FAMILIES <<< "${1#*=}"
      shift
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

for I in "${!FAMILIES[@]}"; do
  FAMILIES[$I]=$(echo "${FAMILIES[$I]}" | tr '[:upper:]' '[:lower:]')
  MATCH=false
  for KF in "${KNOWN_FAMILIES[@]}"; do
    [ "${FAMILIES[$I]}" = "${KF}" ] && MATCH=true && break
  done
  if [ "${MATCH}" = false ]; then
    echo "[ERROR] unknown family '${FAMILIES[$I]}' -- expected one of: ${KNOWN_FAMILIES[*]}" >&2
    exit 1
  fi
done

MODELS_DIR="/projects/med_flan/models/${MAIN_FOLDER}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/infer_eval_multitask.runai.sh"

# Which checkpoint ranks (1-indexed, ascending by step number) to evaluate.
RANKS=(3 5 7 8 9 10)

# Expected checkpoint count per variant -- RANKS above assumes a variant has
# exactly this many checkpoints (rank 10 = the last one). A variant with a
# different count is skipped entirely (not just the out-of-range ranks)
# since its rank-N checkpoints wouldn't line up with every other variant's,
# making cross-variant comparisons at a given rank misleading.
EXPECTED_CHECKPOINTS=10

CHECKING_DIR="/hpcfs/runai_data/projects/genai-gs/med_flan/models/${MAIN_FOLDER}"

if [ ! -d "${CHECKING_DIR}" ]; then
  echo "[ERROR] ${CHECKING_DIR} does not exist" >&2
  exit 1
fi

if [ "${DO_SUBMIT}" = true ]; then
  echo "[INFO] Sweeping ${MODELS_DIR} -- SUBMITTING"
else
  echo "[INFO] Sweeping ${MODELS_DIR} -- DRY RUN (pass --yes to actually submit)"
fi
echo "[INFO] Checkpoint ranks: ${RANKS[*]}"
if [ ${#FAMILIES[@]} -gt 0 ]; then
  echo "[INFO] Restricting to families: ${FAMILIES[*]}"
fi
echo "[INFO] Sampling: temperature=${TEMPERATURE:-0.0} top_p=${TOP_P:-1.0}"

PLANNED=()
FAILED=()

for VARIANT_PATH in "${CHECKING_DIR}"/*/; do
  [ -d "${VARIANT_PATH}" ] || continue
  VARIANT_DIR=$(basename "${VARIANT_PATH}")
  FAMILY=$(echo "${VARIANT_DIR}" | cut -d'_' -f1 | tr '[:upper:]' '[:lower:]')

  if [ ${#FAMILIES[@]} -gt 0 ]; then
    SKIP=true
    for F in "${FAMILIES[@]}"; do
      [ "${F}" = "${FAMILY}" ] && SKIP=false && break
    done
    if [ "${SKIP}" = true ]; then
      continue
    fi
  fi

  # sorted ascending by the checkpoint's numeric step -- variant names use
  # only underscores (never dashes), so splitting the whole path on '-' and
  # taking the last field unambiguously yields the checkpoint number.
  # (while-read into an array instead of `mapfile`, which needs bash 4+ and
  # isn't guaranteed on every login node / macOS's default /bin/bash.)
  CKPT_DIRS=()
  while IFS= read -r line; do
    CKPT_DIRS+=("${line}")
  done < <(
    find "${VARIANT_PATH}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
      | awk -F- '{print $NF, $0}' \
      | sort -n \
      | cut -d' ' -f2-
  )
  N=${#CKPT_DIRS[@]}
  if [ "${N}" -eq 0 ]; then
    echo "[WARN] no checkpoint-* dirs under ${VARIANT_PATH}, skipping" >&2
    continue
  fi
  if [ "${N}" -ne "${EXPECTED_CHECKPOINTS}" ]; then
    echo "[WARN] ${VARIANT_DIR}: found ${N} checkpoint-* dirs, expected ${EXPECTED_CHECKPOINTS} -- skipping variant entirely" >&2
    continue
  fi

  case "${FAMILY}" in
    hrm) IMAGE_LABEL="hrm_train:base" ;;
    ouro|qwen) IMAGE_LABEL="ouro_train:base" ;;
    *) IMAGE_LABEL="UNKNOWN (will fail in infer_eval_multitask.runai.sh)" ;;
  esac

  for RANK in "${RANKS[@]}"; do
    IDX=$((RANK - 1))
    if [ "${IDX}" -lt 0 ] || [ "${IDX}" -ge "${N}" ]; then
      echo "[WARN] ${VARIANT_DIR}: rank ${RANK} out of range (only ${N} checkpoints), skipping" >&2
      continue
    fi
    MODEL_PATH="${MODELS_DIR}${CKPT_DIRS[${IDX}]#${CHECKING_DIR}}"
    CKPT_NAME=$(basename "${MODEL_PATH}")
    echo "[PLAN] ${VARIANT_DIR} rank ${RANK} -> ${CKPT_NAME} (image ${IMAGE_LABEL})"
    PLANNED+=("${MODEL_PATH}|${RANK}")
  done
done

echo "[INFO] ${#PLANNED[@]} job(s) would be submitted for main_folder=${MAIN_FOLDER}"

if [ "${DO_SUBMIT}" = false ]; then
  echo "[INFO] Dry run complete -- re-run with --yes to submit."
  exit 0
fi

for ENTRY in "${PLANNED[@]}"; do
  MODEL_PATH="${ENTRY%%|*}"
  RANK="${ENTRY##*|}"
  echo "[INFO] === submitting ${MODEL_PATH} rank ${RANK} ==="
  if ! bash "${SUBMIT_SCRIPT}" "${MODEL_PATH}" "${RANK}"; then
    echo "[ERROR] submission failed: ${MODEL_PATH} (rank ${RANK})" >&2
    FAILED+=("${MODEL_PATH} (rank ${RANK})")
  fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "[ERROR] ${#FAILED[@]} submission(s) failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi

echo "[INFO] All submissions complete."
