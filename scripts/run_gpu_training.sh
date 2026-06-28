#!/usr/bin/env bash
set -euo pipefail

echo "Install GPU extras first: pip install -r requirements-gpu.txt"
python src/train_t1_gpu.py --allow-cpu false

if [[ -f data/processed/n1_gemma_train.jsonl ]]; then
  python src/train_n1_gemma_lora.py
else
  echo "Skipping Gemma LoRA training because data/processed/n1_gemma_train.jsonl is missing."
fi
