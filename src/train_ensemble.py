from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.config import CANDIDATES_DIR, ENSEMBLE_PRODUCTION_DIR, ensure_project_dirs
from src.feature_store import load_training_frame
from src.device import is_training_gpu_available
from src.modeling import (
    build_recency_weights,
    merged_feature_columns,
    prepare_model_frame,
    regression_metrics,
    save_model_bundle,
    split_timeframe,
    version_stamp,
)
from src.train_t1_gpu import build_xgb_regressor
from src.utils import setup_logging, utc_now_iso


def train_ensemble_model(output_dir: Path | None = None) -> dict:
    ensure_project_dirs()
    if not is_training_gpu_available():
        raise RuntimeError("YOLO-WALLSTREET ensemble training requires NVIDIA CUDA.")

    output_dir = output_dir or ENSEMBLE_PRODUCTION_DIR
    merged = load_training_frame(include_news=True, include_chronos=True)
    feature_columns = merged_feature_columns()
    frame = prepare_model_frame(merged, feature_columns, target_column="future_ret_5d")
    train_df, validation_df, test_df = split_timeframe(frame)

    if train_df.empty or validation_df.empty:
        raise RuntimeError("Insufficient train/validation data for ensemble GPU training.")

    model = build_xgb_regressor()
    train_weights = build_recency_weights(train_df, half_life_days=180)
    validation_weights = build_recency_weights(validation_df, half_life_days=180)
    model.fit(
        train_df[feature_columns],
        train_df["future_ret_5d"],
        sample_weight=train_weights,
        eval_set=[(validation_df[feature_columns], validation_df["future_ret_5d"])],
        sample_weight_eval_set=[validation_weights],
        verbose=100,
    )

    validation_pred = model.predict(validation_df[feature_columns])
    validation_metrics = regression_metrics(validation_df["future_ret_5d"], validation_pred)

    test_metrics = {}
    if not test_df.empty:
        test_pred = model.predict(test_df[feature_columns])
        test_metrics = regression_metrics(test_df["future_ret_5d"], test_pred)

    model_version = version_stamp("YOLO-WALLSTREET-ensemble-gpu")
    metadata = {
        "model_name": "YOLO-WALLSTREET-ensemble",
        "model_version": model_version,
        "trained_at": utc_now_iso(),
        "target": "future_ret_5d",
        "feature_columns": feature_columns,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "uses": {"t1": "YOLO-WALLSTREET-t1", "n1": "YOLO-WALLSTREET-n1"},
        "device_used": "cuda",
        "training_accelerator": "nvidia_cuda",
        "mac_inference_supported": True,
        "implementation_backend": "xgboost_cuda",
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
    }
    save_model_bundle(model, metadata, output_dir, artifact_type="xgboost_json")

    validation_predictions = validation_df[["date", "ticker", "close", "future_ret_5d"]].copy()
    validation_predictions["prediction"] = validation_pred
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    logging.info("Saved ensemble GPU model to %s", output_dir)
    return metadata


def resolve_output_dir(destination: str) -> Path:
    if destination == "production":
        return ENSEMBLE_PRODUCTION_DIR
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return CANDIDATES_DIR / "ensemble" / timestamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", choices=["production", "candidate"], default="production")
    args = parser.parse_args()
    setup_logging()
    metadata = train_ensemble_model(output_dir=resolve_output_dir(args.destination))
    print(metadata["model_version"])


if __name__ == "__main__":
    main()
