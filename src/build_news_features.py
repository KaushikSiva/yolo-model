from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import logging
import argparse
from pathlib import Path

import pandas as pd

from src.config import FEATURES_PATH, FINGPT_EVENT_FEATURES_PATH, NEWS_FEATURES_PATH, N1_PRODUCTION_DIR, ensure_project_dirs
from src.news_ingest import load_news_jsonl_files
from src.structured_llm import generate_structured_json, structured_llm_backend_label, structured_llm_model_name, uses_remote_structured_llm
from src.utils import load_json, save_json, setup_logging


TOPIC_KEYWORDS = {
    "earnings_count": {"earnings", "results"},
    "guidance_count": {"guidance", "outlook"},
    "ai_count": {"ai", "artificial intelligence", "gpu", "model"},
    "lawsuit_count": {"lawsuit", "litigation", "sued"},
    "analyst_count": {"analyst", "rating", "price target", "downgrade", "upgrade"},
    "regulatory_count": {"regulatory", "regulation", "antitrust", "investigation", "sec"},
}

SUPPORTED_BUILD_MODES = {"heuristic", "fingpt", "hybrid"}


def _fallback_payload(text: str) -> dict:
    lowered = text.lower()
    positive_hits = sum(
        word in lowered
        for word in {"beat", "strong", "growth", "upbeat", "surge", "record", "expand", "positive", "upgrade"}
    )
    negative_hits = sum(
        word in lowered
        for word in {"miss", "weak", "cut", "drop", "decline", "lawsuit", "risk", "negative", "downgrade"}
    )
    total_hits = positive_hits + negative_hits
    sentiment_score = 0.0 if total_hits == 0 else float((positive_hits - negative_hits) / total_hits)
    catalyst_type = "general_news"
    for feature_name, keywords in TOPIC_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            catalyst_type = feature_name.replace("_count", "")
            break
    risk_flags = []
    if any(word in lowered for word in {"lawsuit", "litigation", "investigation", "antitrust", "sec"}):
        risk_flags.append("legal_or_regulatory")
    if "guidance" in lowered and any(word in lowered for word in {"cut", "lower", "reduced"}):
        risk_flags.append("guidance_cut")
    return {
        "sentiment": "positive" if sentiment_score > 0.15 else "negative" if sentiment_score < -0.15 else "neutral",
        "sentiment_score": round(sentiment_score, 4),
        "catalyst_type": catalyst_type,
        "company_specific": True,
        "risk_flags": risk_flags,
        "novelty": "medium",
        "likely_horizon": "5_20_trading_days" if any(word in lowered for word in {"earnings", "guidance", "quarter"}) else "1_5_trading_days",
        "confidence": 0.35,
    }


def _normalize_novelty(value: str) -> float:
    mapping = {"low": 0.2, "medium": 0.55, "high": 0.9}
    return mapping.get(str(value).lower(), 0.4)


def _keyword_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        feature_name: int(any(word in lowered for word in keywords))
        for feature_name, keywords in TOPIC_KEYWORDS.items()
    }


def _build_prompt(row: dict) -> str:
    market_context = row.get("market_context", {})
    return (
        "Instruction: Extract stock-relevant trading features as valid JSON.\n"
        "Input: "
        f"Ticker: {row['ticker']}\n"
        f"Date: {pd.Timestamp(row['published_at']).date().isoformat()}\n"
        f"Headline: {row.get('title', '')}\n"
        f"Article: {row.get('body', '')}\n"
        f"Source: {row.get('source', '')}\n"
        f"MarketContext: {json.dumps(market_context, sort_keys=True)}\n"
        "Output: {"
    )


def _load_fingpt_model_dir() -> Path:
    metadata_path = N1_PRODUCTION_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing n1 production metadata: {metadata_path}. Train or export the FinGPT model first."
        )
    metadata = load_json(metadata_path)
    model_dir = Path(metadata.get("artifact_path", N1_PRODUCTION_DIR))
    if not model_dir.exists():
        raise FileNotFoundError(f"FinGPT artifact directory does not exist: {model_dir}")
    return model_dir


def _lookup_market_context(features: pd.DataFrame, ticker: str, published_at: str | pd.Timestamp) -> dict:
    published_day = pd.Timestamp(published_at)
    if published_day.tzinfo is not None:
        published_day = published_day.tz_convert("UTC").tz_localize(None)
    published_day = published_day.normalize()
    rows = features.loc[(features["ticker"] == ticker) & (features["date"] <= published_day)].sort_values("date")
    if rows.empty:
        return {}
    feature_row = rows.iloc[-1]
    return {
        "ret_1d": round(float(feature_row.get("ret_1d", 0.0) or 0.0), 4),
        "ret_5d": round(float(feature_row.get("ret_5d", 0.0) or 0.0), 4),
        "volatility_20d": round(float(feature_row.get("volatility_20d", 0.0) or 0.0), 4),
        "future_ret_1d": round(float(feature_row.get("future_ret_1d", 0.0) or 0.0), 4),
        "future_ret_5d": round(float(feature_row.get("future_ret_5d", 0.0) or 0.0), 4),
        "future_ret_20d": round(float(feature_row.get("future_ret_20d", 0.0) or 0.0), 4),
    }


def _payload_to_feature_row(row: dict, payload: dict, text: str) -> dict:
    published_day = pd.Timestamp(row["published_at"])
    if published_day.tzinfo is not None:
        published_day = published_day.tz_convert("UTC").tz_localize(None)
    topic_counts = _keyword_counts(text)
    sentiment_score = float(payload.get("sentiment_score", 0.0) or 0.0)
    company_specific = 1.0 if bool(payload.get("company_specific", False)) else 0.0
    novelty_score = _normalize_novelty(str(payload.get("novelty", "medium")))
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    risk_flags = payload.get("risk_flags", []) or []
    return {
        "ticker": str(row["ticker"]).upper(),
        "date": published_day.normalize(),
        "news_count": 1.0,
        "avg_sentiment_stub": sentiment_score,
        "max_positive_sentiment_stub": max(sentiment_score, 0.0),
        "max_negative_sentiment_stub": min(sentiment_score, 0.0),
        "company_specific_score": company_specific,
        "macro_relevance_score": 1.0 - company_specific,
        "novelty_score": novelty_score,
        "materiality_score": min(1.0, abs(sentiment_score) * 0.5 + confidence * 0.5),
        "event_confidence_score": confidence,
        "risk_flag_count": len(risk_flags),
        **topic_counts,
    }


def _infer_heuristic_event_rows(news_rows: list[dict]) -> pd.DataFrame:
    extracted_rows: list[dict] = []
    for row in news_rows:
        text = f"{row.get('title', '')} {row.get('body', '')}".strip()
        payload = _fallback_payload(text)
        extracted_rows.append(_payload_to_feature_row(row, payload, text))
    return pd.DataFrame(extracted_rows)


def _infer_fingpt_event_rows(
    model_dir: Path | None,
    features: pd.DataFrame,
    news_rows: list[dict],
    allow_fallback: bool,
) -> pd.DataFrame:
    extracted_rows: list[dict] = []
    for row in news_rows:
        row = dict(row)
        row["market_context"] = _lookup_market_context(features, str(row["ticker"]).upper(), row["published_at"])
        text = f"{row.get('title', '')} {row.get('body', '')}".strip()
        try:
            payload = generate_structured_json(
                model_dir,
                _build_prompt(row),
                max_new_tokens=160,
                min_new_tokens=24,
                json_prefix="{",
            )
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    f"FinGPT structured extraction failed for {row.get('ticker', '')} {row.get('published_at', '')}: {exc}"
                ) from exc
            logging.warning(
                "FinGPT structured extraction failed for %s %s: %s. Falling back to heuristic payload.",
                row.get("ticker", ""),
                row.get("published_at", ""),
                exc,
            )
            payload = _fallback_payload(text)
        extracted_rows.append(_payload_to_feature_row(row, payload, text))
    return pd.DataFrame(extracted_rows)


def build_news_features(mode: str = "heuristic") -> pd.DataFrame:
    ensure_project_dirs()
    mode = str(mode).strip().lower()
    if mode not in SUPPORTED_BUILD_MODES:
        raise ValueError(f"Unsupported build mode: {mode}. Expected one of {sorted(SUPPORTED_BUILD_MODES)}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

    base = pd.read_parquet(FEATURES_PATH)[["ticker", "date"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"])
    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])
    news_rows = load_news_jsonl_files()
    if not news_rows:
        raise FileNotFoundError("No raw news JSONL files found. Run src/news_ingest.py first.")

    if mode == "heuristic":
        inferred = _infer_heuristic_event_rows(news_rows)
    else:
        model_dir = None if uses_remote_structured_llm() else _load_fingpt_model_dir()
        inferred = _infer_fingpt_event_rows(
            model_dir,
            features,
            news_rows,
            allow_fallback=mode == "hybrid",
        )
    if inferred.empty:
        raise RuntimeError(f"News feature extraction produced no rows in mode={mode}.")

    aggregated = inferred.groupby(["ticker", "date"], as_index=False).agg(
        news_count=("news_count", "sum"),
        avg_sentiment_stub=("avg_sentiment_stub", "mean"),
        max_positive_sentiment_stub=("max_positive_sentiment_stub", "max"),
        max_negative_sentiment_stub=("max_negative_sentiment_stub", "min"),
        earnings_count=("earnings_count", "sum"),
        guidance_count=("guidance_count", "sum"),
        ai_count=("ai_count", "sum"),
        lawsuit_count=("lawsuit_count", "sum"),
        analyst_count=("analyst_count", "sum"),
        regulatory_count=("regulatory_count", "sum"),
        company_specific_score=("company_specific_score", "mean"),
        macro_relevance_score=("macro_relevance_score", "mean"),
        novelty_score=("novelty_score", "max"),
        materiality_score=("materiality_score", "max"),
        event_confidence_score=("event_confidence_score", "mean"),
        risk_flag_count=("risk_flag_count", "sum"),
    )
    aggregated.to_parquet(FINGPT_EVENT_FEATURES_PATH, index=False)
    save_json(
        FINGPT_EVENT_FEATURES_PATH.with_suffix(".metadata.json"),
        {"mode": mode, "rows": len(aggregated), "source": "heuristic_rules" if mode == "heuristic" else "fingpt_model_inference"},
    )

    result = base.merge(aggregated, on=["ticker", "date"], how="left").fillna(0.0)
    NEWS_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(NEWS_FEATURES_PATH, index=False)

    metadata = load_json(N1_PRODUCTION_DIR / "metadata.json", default={})
    n1_metadata = {
        "model_name": "YOLO-WALLSTREET-n1",
        "model_version": metadata.get("model_version") or structured_llm_model_name(),
        "artifact_path": metadata.get("artifact_path"),
        "feature_builder_backend": structured_llm_backend_label(),
        "feature_builder_mode": mode,
        "gpu_training_required": mode != "heuristic",
        "mac_inference_supported": False,
        "type": "heuristic_news_feature_extractor" if mode == "heuristic" else "fingpt_structured_feature_extractor",
    }
    save_json(N1_PRODUCTION_DIR / "metadata.json", n1_metadata)
    logging.info("Saved news features to %s with %s rows using mode=%s", NEWS_FEATURES_PATH, len(result), mode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(SUPPORTED_BUILD_MODES), default="heuristic")
    args = parser.parse_args()
    setup_logging()
    build_news_features(mode=args.mode)


if __name__ == "__main__":
    main()
