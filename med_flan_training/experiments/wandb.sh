export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"
export WANDB_CACHE_DIR="/project/log/wandb"
export WANDB_DIR="/project/log/log/wandb"

# Offline by default: set WANDB_MODE=online and provide WANDB_API_KEY/
# WANDB_BASE_URL to enable remote logging. In offline mode, runs log locally;
# sync later with `wandb sync <run_dir>` once connectivity is configured.
export WANDB_MODE="${WANDB_MODE:-offline}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://wandb.example.com}"
export WANDB_USERNAME="${WANDB_USERNAME:-${USER}}"

# export WANDB_API_KEY="REPLACE_ME"
# export WANDB_USERNAME="${WANDB_USERNAME:-${USER}}"
# export WANDB_DISABLE_CODE="true"
# export WANDB_LOG_MODEL="false"
# export WANDB_WATCH="false"
# export WANDB_CACHE_DIR="/project/log/wandb"
# export WANDB_DIR="/project/log/wandb"