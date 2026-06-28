from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from pathlib import Path

import pandas as pd

from src.config import FEATURES_PATH, T1_FEATURE_COLUMNS, T1_PRODUCTION_DIR, ensure_project_dirs
from src.modeling import prepare_model_frame, regression_metrics, save_model_bundle, split_timeframe, version_stamp
from src.utils import setup_logging, utc_now_iso


def build_regressor():
    try:
        from lightgbm import LGBMRegressor

        return (
            LGBMRegressor(
                n_estimators=800,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "lightgbm",
        )
    except Exception as exc:
        logging.warning("Falling back from LightGBM to HistGradientBoostingRegressor: %s", exc)
        from sklearn.ensemble import HistGradientBoostingRegressor

        return (
            HistGradientBoostingRegressor(
                learning_rate=0.02,
                max_depth=5,
                max_iter=800,
                random_state=42,
            ),
            "sklearn_hist_gradient_boosting",
        )


def train_t1_model(output_dir: Path | None = None) -> dict:
    ensure_project_dirs()
    output_dir = output_dir or T1_PRODUCTION_DIR
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    frame = prepare_model_frame(df, T1_FEATURE_COLUMNS, target_column="future_ret_5d")
    train_df, validation_df, test_df = split_timeframe(frame)

    if train_df.empty or validation_df.empty:
        raise RuntimeError("Insufficient train/validation data for t1 baseline.")

    model, backend = build_regressor()
    model.fit(train_df[T1_FEATURE_COLUMNS], train_df["future_ret_5d"])

    validation_pred = model.predict(validation_df[T1_FEATURE_COLUMNS])
    validation_metrics = regression_metrics(validation_df["future_ret_5d"], validation_pred)

    test_metrics = {}
    if not test_df.empty:
        test_pred = model.predict(test_df[T1_FEATURE_COLUMNS])
        test_metrics = regression_metrics(test_df["future_ret_5d"], test_pred)

    model_version = version_stamp("YOLO-WALLSTREET-t1")
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
        "device_used": "cpu",
        "mac_inference_supported": True,
        "implementation_backend": backend,
    }
    save_model_bundle(model, metadata, output_dir)

    validation_predictions = validation_df[["date", "ticker", "close", "future_ret_5d"]].copy()
    validation_predictions["prediction"] = validation_pred
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    logging.info("Saved t1 baseline model to %s", output_dir)
    return metadata


def main() -> None:
    setup_logging()
    metadata = train_t1_model()
    print(metadata["model_version"])


if __name__ == "__main__":
    main()
