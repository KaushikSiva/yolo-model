from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import and_, select, update

from src.db import create_tables, get_engine, predictions_table
from src.utils import setup_logging, utc_now_iso


def fetch_actual_close(ticker: str, target_date: str) -> float | None:
    start = (pd.Timestamp(target_date) - timedelta(days=3)).date().isoformat()
    end = (pd.Timestamp(target_date) + timedelta(days=7)).date().isoformat()
    history = yf.download(ticker, start=start, end=end, auto_adjust=True, actions=False, progress=False, threads=False)
    if history.empty:
        return None
    history = history.reset_index()
    history["Date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    target_ts = pd.Timestamp(target_date)
    eligible = history.loc[history["Date"] >= target_ts]
    if eligible.empty:
        return None
    return float(eligible.iloc[0]["Close"])


def update_outcomes() -> int:
    engine = get_engine()
    create_tables(engine)
    today = pd.Timestamp.today().normalize().date().isoformat()
    query = select(predictions_table).where(
        and_(predictions_table.c.actual_close.is_(None), predictions_table.c.target_date < today)
    )

    updated = 0
    with engine.begin() as connection:
        rows = connection.execute(query).mappings().all()
        for row in rows:
            actual_close = fetch_actual_close(row["ticker"], row["target_date"])
            if actual_close is None:
                logging.info("Skipping %s %s, no actual close available", row["ticker"], row["target_date"])
                continue

            actual_return = (actual_close / row["current_close"]) - 1.0
            absolute_error = abs(actual_close - row["predicted_close"])
            return_error = abs(actual_return - row["predicted_return"])
            direction_correct = int(np.sign(actual_return) == np.sign(row["predicted_return"]))
            within_prediction_range = int(row["bear_case"] <= actual_close <= row["bull_case"])

            connection.execute(
                update(predictions_table)
                .where(predictions_table.c.prediction_id == row["prediction_id"])
                .values(
                    actual_close=actual_close,
                    actual_return=actual_return,
                    absolute_error=absolute_error,
                    return_error=return_error,
                    direction_correct=direction_correct,
                    within_prediction_range=within_prediction_range,
                    scored_at=utc_now_iso(),
                )
            )
            updated += 1
    logging.info("Updated %s prediction outcomes", updated)
    return updated


def main() -> None:
    setup_logging()
    print(update_outcomes())


if __name__ == "__main__":
    main()
