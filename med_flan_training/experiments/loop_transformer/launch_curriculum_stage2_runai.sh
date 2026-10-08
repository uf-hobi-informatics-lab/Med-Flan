#!/usr/bin/env bash
# Submits the curriculum stage2 (full-data continuation) RunAI job (HRM-Text-1B
# bp5). Each *.runai.sh is self-contained (sets its own TRAIN_CONF/JOB_NAME/
# GPU_NUM and calls `runai training submit`), so this just iterates and invokes them with
# a short delay between submissions to avoid hammering the RunAI API.
#
# IMPORTANT -- sequencing: every stage2 yaml's model_name_or_path points at the matching
# stage1 job's output_dir (see recipes/loop_transformer/curriculum/*/sft_*_stage1_curriculum.yaml).
# Do NOT run this launcher until launch_curriculum_stage1_runai.sh has fully completed AND
# you've confirmed each stage1 checkpoint actually exists on disk -- if a stage1 job is
# still running or failed, the matching stage2 job will fail to load model_name_or_path
# (or silently pick up a stale/partial checkpoint).
#
# Usage:
#   bash experiments/loop_transformer/launch_curriculum_stage2_runai.sh            # submit all
#   bash experiments/loop_transformer/launch_curriculum_stage2_runai.sh --dry-run  # print, don't submit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

while IFS= read -r -d '' script; do
  echo "=== ${script#${SCRIPT_DIR}/} ==="
  if $DRY_RUN; then
    echo "[dry-run] bash ${script}"
  else
    bash "${script}"
    sleep 5
  fi
done < <(find "${SCRIPT_DIR}/curriculum" -name "*_stage2_curriculum.runai.sh" -print0 | sort -z)

echo "Done submitting curriculum stage2"
