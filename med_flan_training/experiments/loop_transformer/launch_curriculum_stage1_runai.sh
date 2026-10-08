#!/usr/bin/env bash
# Submits the curriculum stage1 (warmup) RunAI job (HRM-Text-1B bp5). Each
# *.runai.sh is self-contained (sets its own TRAIN_CONF/JOB_NAME/
# GPU_NUM and calls `runai training submit`), so this just iterates and invokes
# them with a short delay between submissions to avoid hammering the RunAI API.
#
# IMPORTANT -- sequencing: this is stage1 ONLY. Stage2 (launch_curriculum_stage2_runai.sh)
# trains from stage1's saved checkpoint (model_name_or_path in each stage2 yaml points at
# stage1's output_dir), so do NOT run the stage2 launcher until every stage1 job below has
# actually finished successfully and its checkpoint exists on disk. Unlike the standalone
# SFT jobs, these two launchers are NOT safe to run back-to-back.
#
# Usage:
#   bash experiments/loop_transformer/launch_curriculum_stage1_runai.sh            # submit all
#   bash experiments/loop_transformer/launch_curriculum_stage1_runai.sh --dry-run  # print, don't submit
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
done < <(find "${SCRIPT_DIR}/curriculum" -name "*_stage1_curriculum.runai.sh" -print0 | sort -z)

echo "Done submitting curriculum stage1"
