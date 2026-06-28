from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import NEWS_FEATURE_COLUMNS, T1_FEATURE_COLUMNS, ensure_project_dirs
from src.utils import save_json


def prepare_model_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "future_ret_5d",
) -> pd.DataFrame:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["ticker", "date", "close", target_column])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def split_timeframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["date"] < pd.Timestamp("2025-07-01")].copy()
    validation = df.loc[(df["date"] >= pd.Timestamp("2025-07-01")) & (df["date"] <= pd.Timestamp("2025-12-31"))].copy()
    test = df.loc[df["date"] >= pd.Timestamp("2026-01-01")].copy()
    return train, validation, test


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "direction_accuracy": float("nan"),
            "top_decile_avg_forward_return": float("nan"),
            "bottom_decile_avg_forward_return": float("nan"),
        }

    order = np.argsort(y_pred)
    decile_size = max(1, int(len(y_pred) * 0.1))
    bottom_idx = order[:decile_size]
    top_idx = order[-decile_size:]

    y_true_np = y_true.to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_np, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_np, y_pred))),
        "r2": float(r2_score(y_true_np, y_pred)),
        "direction_accuracy": float(np.mean(np.sign(y_true_np) == np.sign(y_pred))),
        "top_decile_avg_forward_return": float(np.mean(y_true_np[top_idx])),
        "bottom_decile_avg_forward_return": float(np.mean(y_true_np[bottom_idx])),
    }


def save_model_bundle(
    model: Any,
    metadata: dict[str, Any],
    output_dir: Path,
    model_filename: str = "model.joblib",
) -> None:
    ensure_project_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / model_filename)
    save_json(output_dir / "metadata.json", metadata)


def version_stamp(prefix: str) -> str:
    return f"{prefix}-v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


def merged_feature_columns() -> list[str]:
    return T1_FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS
