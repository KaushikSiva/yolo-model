from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from src.config import CHRONOS_FEATURES_PATH, FEATURES_PATH, T1_CHRONOS_PRODUCTION_DIR
from src.feature_store import load_chronos_features
from src.utils import load_json
from src.utils import summarize_feature_drivers


def predict_t1(ticker: str) -> dict:
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    ticker_rows = df.loc[df["ticker"] == ticker].dropna(subset=["close"]).sort_values("date")
    if ticker_rows.empty:
        raise ValueError(f"No features available for ticker {ticker}")

    row = ticker_rows.iloc[-1]
    current_close = float(row["close"])
    as_of_date = row["date"].date().isoformat()

    metadata_path = T1_CHRONOS_PRODUCTION_DIR / "metadata.json"
    if not CHRONOS_FEATURES_PATH.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Chronos prior is required. Run train_t1_chronos.py and ensure Chronos features are present."
        )
    chronos = load_chronos_features()
    chronos_rows = chronos.loc[(chronos["ticker"] == ticker) & (chronos["date"] <= pd.Timestamp(as_of_date))].sort_values("date")
    if chronos_rows.empty:
        raise ValueError(f"No Chronos features available for ticker {ticker} as of {as_of_date}")

    chronos_row = chronos_rows.iloc[-1]
    metadata = load_json(metadata_path, default={})
    predicted_return = float(chronos_row.get("chronos_pred_ret_5d", 0.0) or 0.0)
    expected_close = current_close * (1.0 + predicted_return)
    feature_columns = metadata.get("feature_columns", [column for column in chronos_row.index if column.startswith("chronos_")])

    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "current_close": current_close,
        "t1_predicted_return": predicted_return,
        "t1_expected_close": expected_close,
        "t1_model_version": metadata["model_version"],
        "t1_backend": metadata.get("implementation_backend", "chronos"),
        "t1_drivers": summarize_feature_drivers(chronos_row, feature_columns, importances=None, limit=5),
    }


def main() -> None:
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_t1(args.ticker.upper()), indent=2))


if __name__ == "__main__":
    main()
