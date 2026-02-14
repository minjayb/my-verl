#!/bin/bash
# Usage: ./train_sft.sh [config-name]
# Example: ./train_sft.sh sft_gsm8k-qwen1.5b
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0

CONFIG_NAME="${1:-sft_gsm8k-qwen1.5b}"
RUN_DIR="$SCRIPT_DIR/outputs/sft_$(date +%d-%m_%H%M)"

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    --config-dir "$SCRIPT_DIR/config" \
    --config-name "$CONFIG_NAME" \
    trainer.default_local_dir="$RUN_DIR"
