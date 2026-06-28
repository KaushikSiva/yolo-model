from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.config import REPORTS_DIR, ensure_project_dirs
from src.db import get_engine, predictions_table
from src.utils import setup_logging


def build_training_dataset() -> pd.DataFrame:
    ensure_project_dirs()
    engine = get_engine()
    with engine.begin() as connection:
        rows = connection.execute(select(predictions_table)).mappings().all()

    df = pd.DataFrame(rows)
    if df.empty:
        output = REPORTS_DIR / "prediction_scorecard.csv"
        pd.DataFrame(columns=["section", "key", "value"]).to_csv(output, index=False)
        return pd.DataFrame()

    scored = df.dropna(subset=["actual_close", "actual_return"]).copy()
    if scored.empty:
        output = REPORTS_DIR / "prediction_scorecard.csv"
        pd.DataFrame(columns=["section", "key", "value"]).to_csv(output, index=False)
        return scored

    records = [
        {"section": "overall", "key": "mae", "value": float(scored["absolute_error"].mean())},
        {"section": "overall", "key": "rmse", "value": float(np.sqrt(np.mean(np.square(scored["predicted_close"] - scored["actual_close"]))))},
        {"section": "overall", "key": "direction_accuracy", "value": float(scored["direction_correct"].mean())},
        {"section": "overall", "key": "range_coverage", "value": float(scored["within_prediction_range"].mean())},
        {"section": "overall", "key": "avg_predicted_return", "value": float(scored["predicted_return"].mean())},
        {"section": "overall", "key": "avg_actual_return", "value": float(scored["actual_return"].mean())},
    ]

    by_ticker = scored.groupby("ticker", as_index=False).agg(
        predictions=("prediction_id", "count"),
        mae=("absolute_error", "mean"),
        direction_accuracy=("direction_correct", "mean"),
        avg_actual_return=("actual_return", "mean"),
    )
    for row in by_ticker.to_dict(orient="records"):
        records.append({"section": "ticker", "key": row["ticker"], "value": row})

    scored["confidence_bucket"] = scored["confidence"].fillna("unknown")
    by_conf = scored.groupby("confidence_bucket", as_index=False).agg(
        predictions=("prediction_id", "count"),
        mae=("absolute_error", "mean"),
        direction_accuracy=("direction_correct", "mean"),
        avg_actual_return=("actual_return", "mean"),
    )
    for row in by_conf.to_dict(orient="records"):
        records.append({"section": "confidence_bucket", "key": row["confidence_bucket"], "value": row})

    report = pd.DataFrame(records)
    report.to_csv(REPORTS_DIR / "prediction_scorecard.csv", index=False)
    return report


def main() -> None:
    setup_logging()
    report = build_training_dataset()
    print(f"Saved scorecard with {len(report)} rows.")


if __name__ == "__main__":
    main()
