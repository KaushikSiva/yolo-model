from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CHRONOS_FEATURES_PATH, CHRONOS_FEATURE_COLUMNS, FEATURES_PATH, FINGPT_EVENT_FEATURES_PATH, NEWS_FEATURES_PATH, NEWS_FEATURE_COLUMNS


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_price_features() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_chronos_features() -> pd.DataFrame:
    frame = _load_optional_parquet(CHRONOS_FEATURES_PATH)
    if frame.empty:
        columns = ["ticker", "date", *CHRONOS_FEATURE_COLUMNS]
        return pd.DataFrame(columns=columns)
    for column in CHRONOS_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[["ticker", "date", *CHRONOS_FEATURE_COLUMNS]].copy()


def load_news_features() -> pd.DataFrame:
    frame = _load_optional_parquet(NEWS_FEATURES_PATH)
    if frame.empty:
        columns = ["ticker", "date", *NEWS_FEATURE_COLUMNS]
        return pd.DataFrame(columns=columns)
    for column in NEWS_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[["ticker", "date", *NEWS_FEATURE_COLUMNS]].copy()


def load_fingpt_event_features() -> pd.DataFrame:
    frame = _load_optional_parquet(FINGPT_EVENT_FEATURES_PATH)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "company_specific_score",
                "macro_relevance_score",
                "novelty_score",
                "materiality_score",
                "event_confidence_score",
                "risk_flag_count",
            ]
        )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_training_frame(include_news: bool = True, include_chronos: bool = False) -> pd.DataFrame:
    frame = load_price_features()
    if include_chronos:
        chronos = load_chronos_features()
        if not chronos.empty:
            frame = frame.merge(chronos, on=["ticker", "date"], how="left")
    if include_news:
        news = load_news_features()
        if not news.empty:
            frame = frame.merge(news, on=["ticker", "date"], how="left")
    return frame


def latest_row_for_ticker(frame: pd.DataFrame, ticker: str) -> pd.Series:
    rows = frame.loc[frame["ticker"] == ticker].sort_values("date")
    if rows.empty:
        raise ValueError(f"No rows available for ticker {ticker}")
    return rows.iloc[-1]
