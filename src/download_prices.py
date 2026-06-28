from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import logging

import pandas as pd
import yfinance as yf

from src.config import FAILED_TICKERS_PATH, OHLCV_PATH, ensure_project_dirs, load_settings
from src.universe import get_category_map, get_tickers
from src.utils import setup_logging


def _normalize_column_name(column: object) -> str:
    if isinstance(column, tuple):
        column = column[0]
    return str(column).strip().lower().replace(" ", "_")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = df.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [column[0] for column in flattened.columns]
    flattened.columns = [_normalize_column_name(column) for column in flattened.columns]
    return flattened


def _manually_auto_adjust(history: pd.DataFrame) -> pd.DataFrame:
    adjusted = history.copy()
    if "adj_close" in adjusted.columns:
        ratio = adjusted["adj_close"] / adjusted["close"].replace(0, pd.NA)
        for column in ["open", "high", "low", "close"]:
            adjusted[column] = adjusted[column] * ratio
    return adjusted


def download_prices() -> pd.DataFrame:
    ensure_project_dirs()
    settings = load_settings()
    tickers = get_tickers(include_benchmarks=True)
    category_map = get_category_map()

    period = settings.get("data", {}).get("price_period", "3y")
    interval = settings.get("data", {}).get("price_interval", "1d")
    auto_adjust = settings.get("data", {}).get("auto_adjust", True)
    actions = settings.get("data", {}).get("actions", False)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        logging.info("Downloading %s", ticker)
        try:
            history = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=actions,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
        except Exception as exc:
            logging.warning("Failed to download %s: %s", ticker, exc)
            failed.append(ticker)
            continue

        if history.empty:
            logging.warning("No price history for %s", ticker)
            failed.append(ticker)
            continue

        history = _flatten_columns(history)
        history = history.reset_index()
        history = _flatten_columns(history)
        if auto_adjust:
            history = _manually_auto_adjust(history)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(history.columns):
            logging.warning("Unexpected schema for %s: %s", ticker, history.columns.tolist())
            failed.append(ticker)
            continue

        frame = history[["date", "open", "high", "low", "close", "volume"]].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame["ticker"] = ticker
        frame["category"] = category_map.get(ticker, "unknown")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No price data downloaded. Check network connectivity or yfinance availability.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    OHLCV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OHLCV_PATH, index=False)
    FAILED_TICKERS_PATH.write_text(json.dumps({"failed_tickers": failed}, indent=2), encoding="utf-8")
    logging.info("Saved OHLCV data to %s with %s rows", OHLCV_PATH, len(combined))
    if failed:
        logging.warning("Failed tickers: %s", ", ".join(failed))
    return combined


def main() -> None:
    setup_logging()
    download_prices()


if __name__ == "__main__":
    main()
