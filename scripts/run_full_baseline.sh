#!/usr/bin/env bash
set -euo pipefail

echo "Mac/local training has been removed."
echo "Use scripts/run_gpu_training.sh on an NVIDIA CUDA machine to train t1, n1, and ensemble."
echo "This script is now inference-only."
python src/init_db.py
python src/predict_ensemble.py --ticker AAPL --horizon 5d --log
