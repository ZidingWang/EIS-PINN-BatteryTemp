#!/usr/bin/env bash
# One-command transfer-learning run (628Ah -> 314Ah).
# Run from the repository root: bash run.sh
set -euo pipefail

CONFIG="${1:-configs/transfer_314Ah.yaml}"
python -m src.train_transfer --config "$CONFIG"
