#!/usr/bin/env bash
set -euo pipefail

python src/download_prices.py
python src/build_features.py
python src/news_ingest.py --days-back 2
python src/build_news_features.py
python src/update_outcomes.py
python src/predict_all_tickers.py
