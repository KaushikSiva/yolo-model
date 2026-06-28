from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from pathlib import Path

import pandas as pd

from src.config import ENSEMBLE_PRODUCTION_DIR, FEATURES_PATH, NEWS_FEATURES_PATH, ensure_project_dirs
from src.modeling import merged_feature_columns, prepare_model_frame, regression_metrics, save_model_bundle, split_timeframe, version_stamp
from src.train_t1_lgbm import build_regressor
from src.utils import setup_logging, utc_now_iso


def train_ensemble_model(output_dir: Path | None = None) -> dict:
    ensure_project_dirs()
    output_dir = output_dir or ENSEMBLE_PRODUCTION_DIR
    if not FEATURES_PATH.exists() or not NEWS_FEATURES_PATH.exists():
        raise FileNotFoundError("Missing features or news features for ensemble training.")

    features = pd.read_parquet(FEATURES_PATH)
    news = pd.read_parquet(NEWS_FEATURES_PATH)
    merged = features.merge(news, on=["ticker", "date"], how="left")
    feature_columns = merged_feature_columns()
    frame = prepare_model_frame(merged, feature_columns, target_column="future_ret_5d")
    train_df, validation_df, test_df = split_timeframe(frame)

    if train_df.empty or validation_df.empty:
        raise RuntimeError("Insufficient train/validation data for ensemble baseline.")

    model, backend = build_regressor()
    model.fit(train_df[feature_columns], train_df["future_ret_5d"])

    validation_pred = model.predict(validation_df[feature_columns])
    validation_metrics = regression_metrics(validation_df["future_ret_5d"], validation_pred)

    test_metrics = {}
    if not test_df.empty:
        test_pred = model.predict(test_df[feature_columns])
        test_metrics = regression_metrics(test_df["future_ret_5d"], test_pred)

    model_version = version_stamp("YOLO-WALLSTREET-ensemble")
    metadata = {
        "model_name": "YOLO-WALLSTREET-ensemble",
        "model_version": model_version,
        "trained_at": utc_now_iso(),
        "target": "future_ret_5d",
        "feature_columns": feature_columns,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "uses": {"t1": "YOLO-WALLSTREET-t1", "n1": "YOLO-WALLSTREET-n1"},
        "mac_inference_supported": True,
        "implementation_backend": backend,
    }
    save_model_bundle(model, metadata, output_dir)

    validation_predictions = validation_df[["date", "ticker", "close", "future_ret_5d"]].copy()
    validation_predictions["prediction"] = validation_pred
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    logging.info("Saved ensemble model to %s", output_dir)
    return metadata


def main() -> None:
    setup_logging()
    metadata = train_ensemble_model()
    print(metadata["model_version"])


if __name__ == "__main__":
    main()
