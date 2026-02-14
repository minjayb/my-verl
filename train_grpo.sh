#!/bin/bash
# Usage: ./train_grpo.sh [config-name]
# Example: ./train_grpo.sh grpo_gsm8k-qwen1.5b
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Multi-GPU: set to all available GPUs (e.g. "0,1,2,3" for 4x4090, "0,1,2,3,4,5" for 6x3090)
# Must match trainer.n_gpus_per_node in config
export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_NAME="${1:-grpo_gsm8k-qwen1.5b}"
RUN_DIR="$SCRIPT_DIR/outputs/grpo_$(date +%d-%m_%H%M)"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    --config-dir "$SCRIPT_DIR/config" \
    --config-name "$CONFIG_NAME" \
    trainer.default_local_dir="$RUN_DIR"
