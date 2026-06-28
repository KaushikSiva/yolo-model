from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PLANNER_PRODUCTION_DIR
from src.structured_llm import generate_structured_json, structured_llm_backend_label, structured_llm_model_name, uses_remote_structured_llm
from src.utils import load_json


REQUIRED_PLAN_KEYS = {
    "should_retrieve",
    "urgency",
    "urgency_score",
    "triggers",
    "suggested_sources",
    "query_terms",
    "notes",
}


def load_planner_metadata() -> dict[str, Any]:
    metadata_path = PLANNER_PRODUCTION_DIR / "metadata.json"
    if uses_remote_structured_llm() and not metadata_path.exists():
        return {"artifact_path": None, "model_version": structured_llm_model_name()}
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing planner metadata: {metadata_path}. Train or export the Gemma planner first."
        )
    return load_json(metadata_path)


def _coerce_plan(payload: dict[str, Any], ticker: str, horizon: str, current_close: float, model_version: str) -> dict[str, Any]:
    missing = REQUIRED_PLAN_KEYS - set(payload)
    if missing:
        raise ValueError(f"Planner output is missing required keys: {sorted(missing)}")
    return {
        "planner_model_version": model_version,
        "planner_backend": structured_llm_backend_label(),
        "should_retrieve": bool(payload["should_retrieve"]),
        "urgency": str(payload["urgency"]),
        "urgency_score": float(payload["urgency_score"]),
        "target_horizon": horizon,
        "ticker": ticker,
        "current_close": round(float(current_close), 4),
        "triggers": list(payload["triggers"]),
        "suggested_sources": list(payload["suggested_sources"]),
        "query_terms": list(payload["query_terms"]),
        "notes": str(payload["notes"]),
    }


def _build_prompt(
    ticker: str,
    feature_row: pd.Series,
    news_features: dict[str, float],
    t1_payload: dict[str, Any],
    horizon: str,
) -> str:
    return (
        "You are a stock-news retrieval planner. Return only valid JSON with keys "
        "should_retrieve, urgency, urgency_score, triggers, suggested_sources, query_terms, notes.\n"
        "Urgency must be one of low, medium, high.\n"
        "Suggested sources should prioritize public company IR, SEC filings, earnings transcripts, and public news pages.\n\n"
        f"Ticker: {ticker}\n"
        f"Horizon: {horizon}\n"
        f"ret_1d: {float(feature_row.get('ret_1d', 0.0) or 0.0):.6f}\n"
        f"ret_5d: {float(feature_row.get('ret_5d', 0.0) or 0.0):.6f}\n"
        f"volatility_20d: {float(feature_row.get('volatility_20d', 0.0) or 0.0):.6f}\n"
        f"rapid_move: {int(feature_row.get('rapid_move', 0) or 0)}\n"
        f"news_features: {news_features}\n"
        f"t1_predicted_return: {float(t1_payload.get('t1_predicted_return', 0.0) or 0.0):.6f}\n"
    )


def plan_retrieval(
    ticker: str,
    feature_row: pd.Series,
    news_features: dict[str, float],
    t1_payload: dict[str, Any],
    horizon: str,
    current_close: float,
) -> dict[str, Any]:
    metadata = load_planner_metadata()
    model_dir = None if uses_remote_structured_llm() else Path(metadata.get("artifact_path", PLANNER_PRODUCTION_DIR))
    if model_dir is not None and not model_dir.exists():
        raise FileNotFoundError(f"Planner artifact directory does not exist: {model_dir}")

    prompt = _build_prompt(ticker, feature_row, news_features, t1_payload, horizon)
    raw_payload = generate_structured_json(model_dir, prompt, max_new_tokens=256)
    return _coerce_plan(raw_payload, ticker, horizon, current_close, metadata["model_version"])
