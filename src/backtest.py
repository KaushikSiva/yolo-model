from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BENCHMARK_TICKERS, ENSEMBLE_PRODUCTION_DIR, FEATURES_PATH, NEWS_FEATURES_PATH, REPORTS_DIR, T1_PRODUCTION_DIR
from src.modeling import merged_feature_columns
from src.utils import save_json, setup_logging


def run_weekly_backtest(
    model,
    feature_columns: list[str],
    df: pd.DataFrame,
    model_name: str,
    model_version: str,
    slippage: float = 0.001,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.loc[~working["ticker"].isin(BENCHMARK_TICKERS)].copy()
    working = working.dropna(subset=["future_ret_5d"])
    if working.empty:
        empty = pd.DataFrame(columns=["date", "portfolio_return", "benchmark_return"])
        return empty, {
            "model_name": model_name,
            "model_version": model_version,
            "periods": 0,
            "total_return": 0.0,
            "benchmark_total_return": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
        }

    working["week"] = working["date"].dt.to_period("W")
    rebalance_dates = working.groupby("week")["date"].min().sort_values().tolist()
    records = []

    for rebalance_date in rebalance_dates:
        slice_df = working.loc[working["date"] == rebalance_date].copy()
        if slice_df.empty:
            continue
        slice_df[feature_columns] = slice_df[feature_columns].fillna(0.0)
        predictions = model.predict(slice_df[feature_columns])
        slice_df["prediction"] = predictions
        picks = slice_df.nlargest(5, "prediction")
        portfolio_return = float(picks["future_ret_5d"].mean() - slippage)

        qqq = df.loc[(df["ticker"] == "QQQ") & (pd.to_datetime(df["date"]) == rebalance_date), "future_ret_5d"]
        benchmark_return = float(qqq.iloc[0]) if not qqq.empty else 0.0
        records.append(
            {
                "date": rebalance_date.date().isoformat(),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "tickers": ",".join(picks["ticker"].tolist()),
            }
        )

    results = pd.DataFrame(records)
    if results.empty:
        metrics = {
            "model_name": model_name,
            "model_version": model_version,
            "periods": 0,
            "total_return": 0.0,
            "benchmark_total_return": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
        }
        return results, metrics

    cumulative = (1.0 + results["portfolio_return"]).cumprod()
    benchmark_cumulative = (1.0 + results["benchmark_return"]).cumprod()
    running_peak = cumulative.cummax()
    drawdown = (cumulative / running_peak) - 1.0
    sharpe_like = 0.0
    if results["portfolio_return"].std(ddof=0) > 0:
        sharpe_like = float(results["portfolio_return"].mean() / results["portfolio_return"].std(ddof=0) * np.sqrt(52))

    metrics = {
        "model_name": model_name,
        "model_version": model_version,
        "periods": int(len(results)),
        "total_return": float(cumulative.iloc[-1] - 1.0),
        "benchmark_total_return": float(benchmark_cumulative.iloc[-1] - 1.0),
        "sharpe_like": sharpe_like,
        "max_drawdown": float(abs(drawdown.min())),
    }

    if save:
        csv_path = REPORTS_DIR / f"backtest_{model_name}_{model_version}.csv"
        json_path = REPORTS_DIR / f"backtest_{model_name}_{model_version}.json"
        results.to_csv(csv_path, index=False)
        save_json(json_path, metrics)
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["t1", "ensemble"], default="ensemble")
    args = parser.parse_args()
    setup_logging()

    if args.model == "ensemble" and (ENSEMBLE_PRODUCTION_DIR / "model.joblib").exists():
        model_dir = ENSEMBLE_PRODUCTION_DIR
        features = pd.read_parquet(FEATURES_PATH).merge(pd.read_parquet(NEWS_FEATURES_PATH), on=["ticker", "date"], how="left")
        feature_columns = merged_feature_columns()
    else:
        model_dir = T1_PRODUCTION_DIR
        features = pd.read_parquet(FEATURES_PATH)
        from src.config import T1_FEATURE_COLUMNS

        feature_columns = T1_FEATURE_COLUMNS

    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "model.joblib")
    _, metrics = run_weekly_backtest(model, feature_columns, features, metadata["model_name"], metadata["model_version"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
