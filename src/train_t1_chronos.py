from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json

import pandas as pd

from src.config import CHRONOS_FEATURES_PATH, CHRONOS_FEATURE_COLUMNS, FEATURES_PATH, T1_CHRONOS_PRODUCTION_DIR, ensure_project_dirs
from src.device import is_training_gpu_available
from src.utils import save_json, setup_logging, utc_now_iso


def _load_pipeline(base_model: str):
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise RuntimeError("chronos-forecasting is required. Install requirements-gpu.txt first.") from exc

    return Chronos2Pipeline.from_pretrained(base_model, device_map="cuda")


def _forecast_for_history(pipeline, history: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    context_df = history.rename(columns={"ticker": "id", "date": "timestamp", "close": "target"})[["id", "timestamp", "target"]]
    return pipeline.predict_df(
        context_df,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )


def _build_features_for_ticker(
    pipeline,
    ticker_frame: pd.DataFrame,
    min_history: int,
    prediction_length: int,
    step: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
    max_index = len(ticker_frame) - 1
    for idx in range(min_history - 1, max_index + 1, step):
        history = ticker_frame.iloc[: idx + 1][["ticker", "date", "close"]].copy()
        pred_df = _forecast_for_history(pipeline, history, prediction_length)
        pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
        current_close = float(ticker_frame.iloc[idx]["close"])
        row = {
            "ticker": ticker_frame.iloc[idx]["ticker"],
            "date": ticker_frame.iloc[idx]["date"],
            "chronos_pred_ret_1d": float(pred_df.iloc[0]["predictions"] / current_close - 1.0),
            "chronos_pred_ret_5d": float(pred_df.iloc[min(4, len(pred_df) - 1)]["predictions"] / current_close - 1.0),
            "chronos_pred_ret_20d": float(pred_df.iloc[min(19, len(pred_df) - 1)]["predictions"] / current_close - 1.0),
            "chronos_bear_ret_5d": float(pred_df.iloc[min(4, len(pred_df) - 1)]["0.1"] / current_close - 1.0),
            "chronos_bull_ret_5d": float(pred_df.iloc[min(4, len(pred_df) - 1)]["0.9"] / current_close - 1.0),
        }
        median = float(pred_df.iloc[min(4, len(pred_df) - 1)]["predictions"])
        lower = float(pred_df.iloc[min(4, len(pred_df) - 1)]["0.1"])
        upper = float(pred_df.iloc[min(4, len(pred_df) - 1)]["0.9"])
        width = abs(upper - lower) / current_close if current_close else 0.0
        row["chronos_confidence_score"] = max(0.15, min(0.95, 1.0 - width))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", *CHRONOS_FEATURE_COLUMNS])

    sampled = pd.DataFrame(rows)
    expanded = ticker_frame[["ticker", "date"]].merge(sampled, on=["ticker", "date"], how="left").sort_values("date")
    expanded[CHRONOS_FEATURE_COLUMNS] = expanded[CHRONOS_FEATURE_COLUMNS].ffill()
    return expanded.dropna(subset=["chronos_pred_ret_5d"]).copy()


def train_t1_chronos(
    base_model: str = "amazon/chronos-2",
    min_history: int = 120,
    prediction_length: int = 20,
    step: int = 5,
) -> dict:
    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    if not is_training_gpu_available():
        raise RuntimeError("Chronos feature generation requires NVIDIA CUDA.")
    if prediction_length < 20:
        raise ValueError("prediction_length must be at least 20.")
    if step < 1:
        raise ValueError("step must be >= 1.")

    pipeline = _load_pipeline(base_model)
    frame = pd.read_parquet(FEATURES_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[["ticker", "date", "close"]].dropna().sort_values(["ticker", "date"])

    outputs = []
    for ticker, ticker_frame in frame.groupby("ticker", sort=True):
        if len(ticker_frame) < min_history:
            continue
        outputs.append(_build_features_for_ticker(pipeline, ticker_frame, min_history, prediction_length, step))

    if not outputs:
        raise RuntimeError("Chronos generation produced no feature rows.")

    chronos_features = pd.concat(outputs, ignore_index=True)
    chronos_features.to_parquet(CHRONOS_FEATURES_PATH, index=False)

    metadata = {
        "model_name": "YOLO-WALLSTREET-t1",
        "model_version": f"YOLO-WALLSTREET-t1-chronos-v{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}",
        "trained_at": utc_now_iso(),
        "base_model": base_model,
        "feature_columns": CHRONOS_FEATURE_COLUMNS,
        "implementation_backend": "chronos_2",
        "generation_step": step,
        "prediction_length": prediction_length,
        "min_history": min_history,
        "mac_inference_supported": True,
    }
    save_json(T1_CHRONOS_PRODUCTION_DIR / "metadata.json", metadata)
    return {"rows": len(chronos_features), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="amazon/chronos-2")
    parser.add_argument("--min-history", type=int, default=120)
    parser.add_argument("--prediction-length", type=int, default=20)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()
    setup_logging()
    print(
        json.dumps(
            train_t1_chronos(
                base_model=args.base_model,
                min_history=args.min_history,
                prediction_length=args.prediction_length,
                step=args.step,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
