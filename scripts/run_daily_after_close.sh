#!/usr/bin/env bash
set -euo pipefail

NEWS_INGEST_MODE="${NEWS_INGEST_MODE:-direct}"

python src/download_prices.py
python src/build_features.py
python src/news_ingest.py --days-back 2 --mode "$NEWS_INGEST_MODE"
python src/build_news_features.py
python src/update_outcomes.py
python src/predict_all_tickers.py
