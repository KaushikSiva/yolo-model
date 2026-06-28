#!/usr/bin/env bash
set -euo pipefail

python src/download_prices.py
python src/build_features.py
python src/news_ingest.py --days-back 7
python src/build_fingpt_training_data.py
python src/train_n1_fingpt.py --destination production
python src/build_news_features.py
python src/build_planner_training_data.py
python src/train_planner_gemma.py --destination production
python src/update_outcomes.py
python src/retrain_candidate.py --model ensemble
python src/evaluate_candidate.py --model ensemble
python src/promote_model.py --model ensemble
