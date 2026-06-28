#!/usr/bin/env bash
set -euo pipefail

python src/download_prices.py
python src/build_features.py
python src/build_news_features.py
python src/update_outcomes.py
python src/retrain_candidate.py --model ensemble
python src/evaluate_candidate.py --model ensemble
python src/promote_model.py --model ensemble
