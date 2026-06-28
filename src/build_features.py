from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging

import numpy as np
import pandas as pd

from src.config import FEATURES_PATH, OHLCV_PATH, ensure_project_dirs
from src.utils import setup_logging


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)

    for window in [1, 3, 5, 10, 20, 60]:
        df[f"ret_{window}d"] = grouped["close"].pct_change(window)

    for window in [5, 20, 60]:
        df[f"volatility_{window}d"] = grouped["close"].pct_change().rolling(window).std().reset_index(level=0, drop=True)

    df["volume_ma20"] = grouped["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    df["volume_std20"] = grouped["volume"].rolling(20).std().reset_index(level=0, drop=True)
    df["volume_z20"] = ((df["volume"] - df["volume_ma20"]) / df["volume_std20"].replace(0, np.nan)).fillna(0.0)
    df["volume_ratio_20"] = (df["volume"] / df["volume_ma20"].replace(0, np.nan)).fillna(0.0)

    for window in [10, 20, 50, 200]:
        df[f"ma{window}"] = grouped["close"].rolling(window).mean().reset_index(level=0, drop=True)
        df[f"close_vs_ma{window}"] = ((df["close"] / df[f"ma{window}"]) - 1.0).replace([np.inf, -np.inf], np.nan)

    df["high_low_range"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["close_open_return"] = ((df["close"] / df["open"]) - 1.0).replace([np.inf, -np.inf], np.nan)

    df["rolling_high_20"] = grouped["high"].rolling(20).max().reset_index(level=0, drop=True)
    df["rolling_low_20"] = grouped["low"].rolling(20).min().reset_index(level=0, drop=True)
    df["close_vs_20d_high"] = ((df["close"] / df["rolling_high_20"]) - 1.0).replace([np.inf, -np.inf], np.nan)
    df["close_vs_20d_low"] = ((df["close"] / df["rolling_low_20"]) - 1.0).replace([np.inf, -np.inf], np.nan)
    df["rapid_move"] = (df["ret_1d"].abs() > 2.5 * df["volatility_20d"]).astype(int)

    for horizon in [1, 5, 10, 20]:
        df[f"future_close_{horizon}d"] = grouped["close"].shift(-horizon)
        df[f"future_ret_{horizon}d"] = (df[f"future_close_{horizon}d"] / df["close"]) - 1.0

    df["future_ret_5d_bucket"] = pd.cut(
        df["future_ret_5d"],
        bins=[-np.inf, -0.05, -0.01, 0.01, 0.05, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype("float")

    return df


def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    benchmarks = {}
    for benchmark in ["QQQ", "SPY"]:
        benchmark_rows = df.loc[df["ticker"] == benchmark, ["date", "ret_1d", "ret_5d", "ret_20d"]].copy()
        benchmark_rows = benchmark_rows.rename(
            columns={
                "ret_1d": f"{benchmark.lower()}_ret_1d",
                "ret_5d": f"{benchmark.lower()}_ret_5d",
                "ret_20d": f"{benchmark.lower()}_ret_20d",
            }
        )
        benchmarks[benchmark] = benchmark_rows

    merged = df.merge(benchmarks["QQQ"], on="date", how="left").merge(benchmarks["SPY"], on="date", how="left")
    merged["stock_minus_qqq_ret_5d"] = merged["ret_5d"] - merged["qqq_ret_5d"]
    merged["stock_minus_spy_ret_5d"] = merged["ret_5d"] - merged["spy_ret_5d"]
    return merged


def build_features() -> pd.DataFrame:
    ensure_project_dirs()
    if not OHLCV_PATH.exists():
        raise FileNotFoundError(f"Missing price file: {OHLCV_PATH}")

    df = pd.read_parquet(OHLCV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    featured = add_group_features(df)
    featured = add_market_context(featured)
    featured = featured.sort_values(["ticker", "date"]).reset_index(drop=True)
    featured = featured.replace([np.inf, -np.inf], np.nan)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(FEATURES_PATH, index=False)
    logging.info("Saved engineered features to %s with %s rows", FEATURES_PATH, len(featured))
    return featured


def main() -> None:
    setup_logging()
    build_features()


if __name__ == "__main__":
    main()
