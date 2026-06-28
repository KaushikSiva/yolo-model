from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json

import joblib
import pandas as pd

from src.config import FEATURES_PATH, T1_PRODUCTION_DIR
from src.utils import summarize_feature_drivers


def predict_t1(ticker: str) -> dict:
    metadata = json.loads((T1_PRODUCTION_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(T1_PRODUCTION_DIR / "model.joblib")
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    ticker_rows = df.loc[df["ticker"] == ticker].dropna(subset=["close"]).sort_values("date")
    if ticker_rows.empty:
        raise ValueError(f"No features available for ticker {ticker}")

    row = ticker_rows.iloc[-1]
    feature_columns = metadata["feature_columns"]
    feature_frame = pd.DataFrame([{column: float(row.get(column, 0.0) or 0.0) for column in feature_columns}])
    predicted_return = float(model.predict(feature_frame)[0])
    current_close = float(row["close"])
    expected_close = current_close * (1.0 + predicted_return)
    importances = getattr(model, "feature_importances_", None)

    return {
        "ticker": ticker,
        "as_of_date": row["date"].date().isoformat(),
        "current_close": current_close,
        "t1_predicted_return": predicted_return,
        "t1_expected_close": expected_close,
        "t1_model_version": metadata["model_version"],
        "t1_drivers": summarize_feature_drivers(row, feature_columns, importances=importances, limit=5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_t1(args.ticker.upper()), indent=2))


if __name__ == "__main__":
    main()
