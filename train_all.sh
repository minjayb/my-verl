#!/bin/bash
# Runs SFT (train + test) sequentially.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting SFT training (train set) ==="
"$SCRIPT_DIR/train_sft.sh"

echo "=== Starting SFT training (test set) ==="
"$SCRIPT_DIR/train_sft-gsm8k-test.sh"

echo "=== All training complete ==="
