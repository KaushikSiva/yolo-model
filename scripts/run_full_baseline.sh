#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Mac/local training has been removed."
echo "Use scripts/run_gpu_training.sh on an NVIDIA CUDA machine to train t1, n1, and ensemble."
echo "This script is now inference-only."
"$PYTHON_BIN" src/init_db.py
"$PYTHON_BIN" src/predict_ensemble.py --ticker AAPL --horizon 5d --log
