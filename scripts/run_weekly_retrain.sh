#!/usr/bin/env bash
set -euo pipefail

NEWS_INGEST_MODE="${NEWS_INGEST_MODE:-direct}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" src/download_prices.py
"$PYTHON_BIN" src/build_features.py
"$PYTHON_BIN" src/news_ingest.py --days-back 7 --mode "$NEWS_INGEST_MODE"
"$PYTHON_BIN" src/build_fingpt_training_data.py
"$PYTHON_BIN" src/train_n1_fingpt.py --destination production
"$PYTHON_BIN" src/build_news_features.py
"$PYTHON_BIN" src/update_outcomes.py
"$PYTHON_BIN" src/retrain_candidate.py --model ensemble
"$PYTHON_BIN" src/evaluate_candidate.py --model ensemble
"$PYTHON_BIN" src/promote_model.py --model ensemble
"$PYTHON_BIN" src/build_adjuster_training_data.py
"$PYTHON_BIN" src/train_adjuster_gemma.py --destination production
