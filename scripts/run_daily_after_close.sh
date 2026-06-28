#!/usr/bin/env bash
set -euo pipefail

NEWS_INGEST_MODE="${NEWS_INGEST_MODE:-direct}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" src/download_prices.py
"$PYTHON_BIN" src/build_features.py
"$PYTHON_BIN" src/news_ingest.py --days-back 2 --mode "$NEWS_INGEST_MODE"
"$PYTHON_BIN" src/build_news_features.py
"$PYTHON_BIN" src/update_outcomes.py
"$PYTHON_BIN" src/predict_all_tickers.py
