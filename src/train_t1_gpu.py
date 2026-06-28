from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import CANDIDATES_DIR, FEATURES_PATH, T1_FEATURE_COLUMNS, T1_PRODUCTION_DIR, ensure_project_dirs
from src.device import is_training_gpu_available
from src.modeling import (
    build_recency_weights,
    prepare_model_frame,
    regression_metrics,
    save_model_bundle,
    split_timeframe,
    version_stamp,
)
from src.utils import setup_logging, utc_now_iso


def build_xgb_regressor():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed. Install requirements-gpu.txt first.") from exc

    return xgb.XGBRegressor(
        n_estimators=2400,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        random_state=42,
        early_stopping_rounds=150,
    )


def train_t1_gpu(output_dir: Path | None = None) -> dict:
    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    if not is_training_gpu_available():
        raise RuntimeError("YOLO-WALLSTREET-t1 GPU training requires NVIDIA CUDA.")

    df = pd.read_parquet(FEATURES_PATH)
    frame = prepare_model_frame(df, T1_FEATURE_COLUMNS, target_column="future_ret_5d")
    train_df, validation_df, test_df = split_timeframe(frame)
    if train_df.empty or validation_df.empty:
        raise RuntimeError("Insufficient train/validation data for t1 GPU training.")

    output_dir = output_dir or T1_PRODUCTION_DIR
    model = build_xgb_regressor()
    train_weights = build_recency_weights(train_df, half_life_days=180)
    validation_weights = build_recency_weights(validation_df, half_life_days=180)

    model.fit(
        train_df[T1_FEATURE_COLUMNS],
        train_df["future_ret_5d"],
        sample_weight=train_weights,
        eval_set=[(validation_df[T1_FEATURE_COLUMNS], validation_df["future_ret_5d"])],
        sample_weight_eval_set=[validation_weights],
        verbose=100,
    )

    validation_pred = model.predict(validation_df[T1_FEATURE_COLUMNS])
    validation_metrics = regression_metrics(validation_df["future_ret_5d"], validation_pred)

    test_metrics = {}
    if not test_df.empty:
        test_pred = model.predict(test_df[T1_FEATURE_COLUMNS])
        test_metrics = regression_metrics(test_df["future_ret_5d"], test_pred)

    model_version = version_stamp("YOLO-WALLSTREET-t1-gpu")
    metadata = {
        "model_name": "YOLO-WALLSTREET-t1",
        "model_version": model_version,
        "trained_at": utc_now_iso(),
        "target": "future_ret_5d",
        "feature_columns": T1_FEATURE_COLUMNS,
        "train_start": train_df["date"].min().date().isoformat(),
        "train_end": train_df["date"].max().date().isoformat(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
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
    logging.info("Saved t1 GPU model to %s", output_dir)
    return metadata


def resolve_output_dir(destination: str) -> Path:
    if destination == "production":
        return T1_PRODUCTION_DIR
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return CANDIDATES_DIR / "t1" / timestamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", choices=["production", "candidate"], default="production")
    args = parser.parse_args()
    setup_logging()
    metadata = train_t1_gpu(output_dir=resolve_output_dir(args.destination))
    print(metadata["model_version"])


if __name__ == "__main__":
    main()
