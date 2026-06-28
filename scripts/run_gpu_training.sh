#!/usr/bin/env bash
set -euo pipefail

NEWS_INGEST_MODE="${NEWS_INGEST_MODE:-direct}"

echo "Install GPU extras first: pip install -r requirements-gpu.txt"
python src/download_prices.py
python src/build_features.py
python src/news_ingest.py --days-back 30 --mode "$NEWS_INGEST_MODE"
python src/train_t1_chronos.py
python src/build_fingpt_training_data.py
python src/train_t1_gpu.py --destination production
python src/train_n1_fingpt.py --destination production
python src/build_news_features.py
python src/build_planner_training_data.py
python src/train_planner_gemma.py --destination production
python src/train_ensemble.py --destination production
