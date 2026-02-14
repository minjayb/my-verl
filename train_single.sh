#!/bin/bash
# Single-example RLVR loop for debugging/experimentation.
# Usage: ./train_single.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0

RUN_DIR="$SCRIPT_DIR/outputs/grpo-single_$(date +%d-%m_%H%M)"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    --config-dir "$SCRIPT_DIR/config" \
    --config-name grpo_single \
    trainer.default_local_dir="$RUN_DIR"
