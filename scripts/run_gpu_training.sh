#!/usr/bin/env bash
set -euo pipefail

NEWS_INGEST_MODE="${NEWS_INGEST_MODE:-direct}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Install GPU extras first: pip install -r requirements-gpu.txt"
"$PYTHON_BIN" src/download_prices.py
"$PYTHON_BIN" src/build_features.py
"$PYTHON_BIN" src/news_ingest.py --days-back 30 --mode "$NEWS_INGEST_MODE"
"$PYTHON_BIN" src/train_t1_chronos.py
"$PYTHON_BIN" src/build_fingpt_training_data.py
"$PYTHON_BIN" src/train_t1_gpu.py --destination production
"$PYTHON_BIN" src/train_n1_fingpt.py --destination production
"$PYTHON_BIN" src/build_news_features.py
"$PYTHON_BIN" src/build_planner_training_data.py
"$PYTHON_BIN" src/train_planner_gemma.py --destination production
"$PYTHON_BIN" src/train_ensemble.py --destination production
