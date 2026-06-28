from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json

import pandas as pd

from src.config import NEWS_FEATURES_PATH, NEWS_FEATURE_COLUMNS, N1_PRODUCTION_DIR
from src.utils import load_json


def predict_n1(ticker: str, as_of_date: str | None = None) -> dict:
    metadata_path = N1_PRODUCTION_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing n1 metadata: {metadata_path}")
    metadata = load_json(metadata_path)
    if not NEWS_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing built news features: {NEWS_FEATURES_PATH}. Run build_news_features.py first."
        )
    df = pd.read_parquet(NEWS_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    ticker_rows = df.loc[df["ticker"] == ticker].sort_values("date")
    if as_of_date:
        ticker_rows = ticker_rows.loc[ticker_rows["date"] <= pd.Timestamp(as_of_date)]
    if not ticker_rows.empty:
        row = ticker_rows.iloc[-1]
        features = {column: float(row.get(column, 0.0) or 0.0) for column in NEWS_FEATURE_COLUMNS}
        as_of = row["date"].date().isoformat()
    else:
        raise ValueError(f"No FinGPT news features available for ticker {ticker} as of {as_of_date}")

    return {
        "ticker": ticker,
        "as_of_date": as_of,
        "n1_model_version": metadata["model_version"],
        "news_features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date")
    args = parser.parse_args()
    print(json.dumps(predict_n1(args.ticker.upper(), args.date), indent=2))


if __name__ == "__main__":
    main()
