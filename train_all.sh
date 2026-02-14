#!/bin/bash
# Runs GRPO then SFT training sequentially.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting SFT training ==="
"$SCRIPT_DIR/train_sft.sh"


echo "=== Starting single-example GRPO (GSM8K) ==="
"$SCRIPT_DIR/train_single.sh"

echo "=== Starting single-example GRPO (DSR) ==="
"$SCRIPT_DIR/train_single_dsr.sh"

echo "=== Starting GRPO training ==="
"$SCRIPT_DIR/train_grpo.sh"

echo "=== All training complete ==="
