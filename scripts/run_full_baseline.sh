#!/usr/bin/env bash
set -euo pipefail

python src/download_prices.py
python src/build_features.py
python src/build_news_features.py
python src/train_t1_lgbm.py
python src/train_ensemble.py
python src/init_db.py
python src/predict_ensemble.py --ticker AAPL --horizon 5d --log
