#!/usr/bin/env bash
# Submits all SFT RunAI jobs across the three model families (hrm_1b, ouro,
# qwen). Each *.runai.sh is self-contained (sets its own TRAIN_CONF/JOB_NAME/
# GPU_NUM and calls `runai training submit`), so this just iterates and invokes
# them with a short delay between submissions to avoid hammering the RunAI API.
#
# Usage:
#   bash experiments/loop_transformer/launch_v1_runai.sh            # submit all
#   bash experiments/loop_transformer/launch_v1_runai.sh --dry-run  # print, don't submit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

ROOTS=("${SCRIPT_DIR}/hrm_1b" "${SCRIPT_DIR}/ouro" "${SCRIPT_DIR}/qwen")

while IFS= read -r -d '' script; do
  echo "=== ${script#${SCRIPT_DIR}/} ==="
  if $DRY_RUN; then
    echo "[dry-run] bash ${script}"
  else
    bash "${script}"
    sleep 5
  fi
done < <(find "${ROOTS[@]}" -name "*.runai.sh" -print0 | sort -z)

echo "Done submitting SFT (hrm_1b + ouro + qwen)"
