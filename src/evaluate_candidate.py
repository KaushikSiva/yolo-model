from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import pandas as pd

from src.backtest import run_weekly_backtest
from src.config import CANDIDATES_DIR, REPORTS_DIR, T1_PRODUCTION_DIR, ENSEMBLE_PRODUCTION_DIR
from src.feature_store import load_training_frame
from src.modeling import load_metadata, load_model_bundle, merged_feature_columns, prepare_model_frame, regression_metrics, split_timeframe
from src.utils import save_json, setup_logging


def _latest_candidate_dir(model_type: str) -> Path:
    candidates = sorted((CANDIDATES_DIR / model_type).glob("*"))
    if not candidates:
        raise FileNotFoundError(f"No candidate directories found for {model_type}")
    return candidates[-1]


def evaluate_candidate_model(model_type: str) -> dict:
    candidate_dir = _latest_candidate_dir(model_type)
    production_dir = T1_PRODUCTION_DIR if model_type == "t1" else ENSEMBLE_PRODUCTION_DIR
    candidate_metadata = load_metadata(candidate_dir)
    production_metadata = load_metadata(production_dir)
    candidate_model, _ = load_model_bundle(candidate_dir)
    production_model, _ = load_model_bundle(production_dir)

    if model_type == "ensemble":
        df = load_training_frame(include_news=True, include_chronos=True)
        feature_columns = merged_feature_columns()
    else:
        df = load_training_frame(include_news=False, include_chronos=False)
        feature_columns = production_metadata["feature_columns"]

    frame = prepare_model_frame(df, feature_columns, target_column="future_ret_5d")
    _, validation_df, _ = split_timeframe(frame)
    validation_df[feature_columns] = validation_df[feature_columns].fillna(0.0)

    candidate_pred = candidate_model.predict(validation_df[feature_columns])
    production_pred = production_model.predict(validation_df[feature_columns])

    candidate_metrics = regression_metrics(validation_df["future_ret_5d"], candidate_pred)
    production_metrics = regression_metrics(validation_df["future_ret_5d"], production_pred)

    _, candidate_backtest = run_weekly_backtest(candidate_model, feature_columns, df, model_type, candidate_metadata["model_version"], save=False)
    _, production_backtest = run_weekly_backtest(production_model, feature_columns, df, model_type, production_metadata["model_version"], save=False)

    evaluation = {
        "model_type": model_type,
        "candidate_dir": str(candidate_dir),
        "candidate_version": candidate_metadata["model_version"],
        "production_version": production_metadata["model_version"],
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
        "candidate_backtest": candidate_backtest,
        "production_backtest": production_backtest,
    }
    save_json(REPORTS_DIR / f"candidate_eval_{model_type}.json", evaluation)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["t1", "ensemble"], required=True)
    args = parser.parse_args()
    setup_logging()
    print(json.dumps(evaluate_candidate_model(args.model), indent=2))


if __name__ == "__main__":
    main()
