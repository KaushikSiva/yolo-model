from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ADJUSTER_PRODUCTION_DIR
from src.news_ingest import brightdata_news_available, fetch_live_news_for_ticker, load_news_jsonl_files
from src.structured_llm import generate_structured_json, structured_llm_backend_label, structured_llm_model_name, uses_remote_structured_llm
from src.utils import clamp, load_json


MAX_ADJUSTMENT_BPS = 150
LOOKBACK_DAYS = 7
MAX_NEWS_ITEMS = 4
REQUIRED_ADJUSTMENT_KEYS = {
    "adjustment_bps",
    "confidence",
    "rationale",
    "cited_signals",
    "risk_flags",
}


def _adjuster_disabled() -> bool:
    value = os.getenv("YOLO_WALLSTREET_DISABLE_ADJUSTER", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def load_adjuster_metadata() -> dict[str, Any]:
    metadata_path = ADJUSTER_PRODUCTION_DIR / "metadata.json"
    if uses_remote_structured_llm() and not metadata_path.exists():
        return {"artifact_path": None, "model_version": structured_llm_model_name()}
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing adjuster metadata: {metadata_path}. Train or export the Gemma adjuster first."
        )
    return load_json(metadata_path)


def _load_recent_news_rows(ticker: str, as_of_date: str, lookback_days: int = LOOKBACK_DAYS, limit: int = MAX_NEWS_ITEMS) -> list[dict]:
    if brightdata_news_available():
        try:
            rows = fetch_live_news_for_ticker(ticker, days_back=lookback_days, max_items=limit, mode="brightdata_api")
            live_rows: list[dict] = []
            for row in rows:
                live_rows.append(
                    {
                        "published_at": str(row.get("published_at", "")).strip(),
                        "title": str(row.get("title", "")).strip(),
                        "source": str(row.get("source", "")).strip(),
                        "body_excerpt": str(row.get("body", "")).strip()[:400],
                        "url": str(row.get("url", "")).strip(),
                    }
                )
            live_rows.sort(key=lambda item: item["published_at"], reverse=True)
            return live_rows[:limit]
        except Exception as exc:
            logging.warning("Bright Data live news fetch failed for %s: %s. Falling back to local cached news.", ticker, exc)

    as_of = pd.Timestamp(as_of_date).normalize()
    earliest = as_of - pd.Timedelta(days=lookback_days)
    rows: list[dict] = []
    for row in load_news_jsonl_files():
        if str(row.get("ticker", "")).upper() != ticker.upper():
            continue
        published_at = pd.Timestamp(row["published_at"]).tz_convert("UTC") if pd.Timestamp(row["published_at"]).tzinfo else pd.Timestamp(row["published_at"], tz="UTC")
        published_day = published_at.tz_convert(None).normalize()
        if earliest <= published_day <= as_of:
            rows.append(
                {
                    "published_at": published_at.isoformat(),
                    "title": str(row.get("title", "")).strip(),
                    "source": str(row.get("source", "")).strip(),
                    "body_excerpt": str(row.get("body", "")).strip()[:400],
                    "url": str(row.get("url", "")).strip(),
                }
            )
    rows.sort(key=lambda item: item["published_at"], reverse=True)
    return rows[:limit]


def _default_adjustment(
    ticker: str,
    horizon: str,
    baseline_predicted_return: float,
    current_close: float,
    model_version: str,
    note: str,
) -> dict[str, Any]:
    return {
        "adjuster_model_version": model_version,
        "adjuster_backend": structured_llm_backend_label() if model_version != "disabled" else "disabled",
        "ticker": ticker,
        "target_horizon": horizon,
        "baseline_predicted_return": round(baseline_predicted_return, 6),
        "baseline_expected_close": round(current_close * (1.0 + baseline_predicted_return), 4),
        "adjustment_bps": 0,
        "confidence": 0.0,
        "rationale": note,
        "cited_signals": [],
        "risk_flags": [],
        "sources_used": [],
        "recent_news": [],
    }


def _coerce_adjustment(
    payload: dict[str, Any],
    ticker: str,
    horizon: str,
    baseline_predicted_return: float,
    current_close: float,
    model_version: str,
    recent_news: list[dict],
) -> dict[str, Any]:
    missing = REQUIRED_ADJUSTMENT_KEYS - set(payload)
    if missing:
        raise ValueError(f"Adjuster output is missing required keys: {sorted(missing)}")
    raw_bps = float(payload["adjustment_bps"])
    adjustment_bps = int(round(clamp(raw_bps, -MAX_ADJUSTMENT_BPS, MAX_ADJUSTMENT_BPS)))
    return {
        "adjuster_model_version": model_version,
        "adjuster_backend": structured_llm_backend_label(),
        "ticker": ticker,
        "target_horizon": horizon,
        "baseline_predicted_return": round(baseline_predicted_return, 6),
        "baseline_expected_close": round(current_close * (1.0 + baseline_predicted_return), 4),
        "adjustment_bps": adjustment_bps,
        "confidence": round(clamp(float(payload["confidence"]), 0.0, 1.0), 4),
        "rationale": str(payload["rationale"]).strip(),
        "cited_signals": [str(value) for value in list(payload["cited_signals"])[:6]],
        "risk_flags": [str(value) for value in list(payload["risk_flags"])[:6]],
        "sources_used": [item["source"] for item in recent_news if item.get("source")],
        "recent_news": recent_news,
    }


def _build_prompt(
    ticker: str,
    horizon: str,
    current_close: float,
    baseline_predicted_return: float,
    volatility_20d: float,
    baseline_risk_flags: list[str],
    news_features: dict[str, float],
    recent_news: list[dict],
) -> str:
    baseline_expected_close = current_close * (1.0 + baseline_predicted_return)
    return (
        "You are a stock prediction adjuster. Read the baseline numeric forecast plus recent stock news and return only valid JSON "
        "with keys adjustment_bps, confidence, rationale, cited_signals, risk_flags.\n"
        f"Keep adjustment_bps as an integer within [-{MAX_ADJUSTMENT_BPS}, {MAX_ADJUSTMENT_BPS}]. "
        "Use 0 when the news is stale, low-signal, or does not justify a change.\n\n"
        f"Ticker: {ticker}\n"
        f"Horizon: {horizon}\n"
        f"Current close: {current_close:.4f}\n"
        f"Baseline predicted_return: {baseline_predicted_return:.6f}\n"
        f"Baseline expected_close: {baseline_expected_close:.4f}\n"
        f"Volatility_20d: {volatility_20d:.6f}\n"
        f"Baseline risk flags: {baseline_risk_flags}\n"
        f"Structured news features: {news_features}\n"
        f"Recent news items: {recent_news}\n"
    )


def adjust_prediction(
    ticker: str,
    as_of_date: str,
    horizon: str,
    current_close: float,
    baseline_predicted_return: float,
    volatility_20d: float,
    baseline_risk_flags: list[str],
    news_features: dict[str, float],
) -> dict[str, Any]:
    if _adjuster_disabled():
        return _default_adjustment(
            ticker=ticker,
            horizon=horizon,
            baseline_predicted_return=baseline_predicted_return,
            current_close=current_close,
            model_version="disabled",
            note="The Gemma adjuster is disabled for this environment, so the baseline forecast was returned unchanged.",
        )

    metadata = load_adjuster_metadata()
    model_dir = None if uses_remote_structured_llm() else Path(metadata.get("artifact_path", ADJUSTER_PRODUCTION_DIR))
    if model_dir is not None and not model_dir.exists():
        raise FileNotFoundError(f"Adjuster artifact directory does not exist: {model_dir}")

    recent_news = _load_recent_news_rows(ticker, as_of_date)
    if not recent_news:
        return _default_adjustment(
            ticker=ticker,
            horizon=horizon,
            baseline_predicted_return=baseline_predicted_return,
            current_close=current_close,
            model_version=metadata["model_version"],
            note="No fresh raw news was available, so the Gemma adjuster applied no change.",
        )

    prompt = _build_prompt(
        ticker=ticker,
        horizon=horizon,
        current_close=current_close,
        baseline_predicted_return=baseline_predicted_return,
        volatility_20d=volatility_20d,
        baseline_risk_flags=baseline_risk_flags,
        news_features=news_features,
        recent_news=recent_news,
    )
    raw_payload = generate_structured_json(model_dir, prompt, max_new_tokens=256)
    return _coerce_adjustment(
        raw_payload,
        ticker=ticker,
        horizon=horizon,
        baseline_predicted_return=baseline_predicted_return,
        current_close=current_close,
        model_version=metadata["model_version"],
        recent_news=recent_news,
    )
