from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def today_iso() -> str:
    return date.today().isoformat()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def parse_horizon(horizon: str) -> int:
    cleaned = horizon.strip().lower()
    if not cleaned.endswith("d"):
        raise ValueError(f"Unsupported horizon: {horizon}")
    return int(cleaned[:-1])


def business_day_offset(as_of_date: str | pd.Timestamp, horizon: str) -> str:
    days = parse_horizon(horizon)
    timestamp = pd.Timestamp(as_of_date)
    return (timestamp + pd.offsets.BDay(days)).date().isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(to_jsonable(data), sort_keys=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.45:
        return "medium-low"
    return "low"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def summarize_feature_drivers(
    row: pd.Series,
    feature_columns: list[str],
    importances: np.ndarray | None = None,
    limit: int = 5,
) -> list[str]:
    scores: list[tuple[str, float, float]] = []
    for idx, column in enumerate(feature_columns):
        value = float(row.get(column, 0.0) or 0.0)
        weight = float(importances[idx]) if importances is not None and idx < len(importances) else 1.0
        scores.append((column, abs(value) * max(weight, 1e-8), value))
    top = sorted(scores, key=lambda item: item[1], reverse=True)[:limit]
    return [f"{column}={value:.4f}" for column, _, value in top]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
