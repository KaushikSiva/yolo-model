from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import FEATURES_PATH, FINGPT_EVENT_FEATURES_PATH, NEWS_FEATURES_PATH, N1_PRODUCTION_DIR, ensure_project_dirs
from src.news_ingest import load_news_jsonl_files
from src.structured_llm import generate_structured_json
from src.utils import load_json, save_json, setup_logging


TOPIC_KEYWORDS = {
    "earnings_count": {"earnings", "results"},
    "guidance_count": {"guidance", "outlook"},
    "ai_count": {"ai", "artificial intelligence", "gpu", "model"},
    "lawsuit_count": {"lawsuit", "litigation", "sued"},
    "analyst_count": {"analyst", "rating", "price target", "downgrade", "upgrade"},
    "regulatory_count": {"regulatory", "regulation", "antitrust", "investigation", "sec"},
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
    return (
        "Extract stock-relevant trading features as valid JSON. "
        "Return only valid JSON with keys sentiment, sentiment_score, catalyst_type, "
        "company_specific, risk_flags, novelty, likely_horizon, confidence.\n\n"
        f"Ticker: {row['ticker']}\n"
        f"Date: {pd.Timestamp(row['published_at']).date().isoformat()}\n"
        f"Headline: {row.get('title', '')}\n"
        f"Article: {row.get('body', '')}\n"
        f"Source: {row.get('source', '')}\n"
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


def _infer_fingpt_event_rows(model_dir: Path, news_rows: list[dict]) -> pd.DataFrame:
    extracted_rows: list[dict] = []
    for row in news_rows:
        payload = generate_structured_json(model_dir, _build_prompt(row), max_new_tokens=256)
        text = f"{row.get('title', '')} {row.get('body', '')}".strip()
        topic_counts = _keyword_counts(text)
        sentiment_score = float(payload.get("sentiment_score", 0.0) or 0.0)
        company_specific = 1.0 if bool(payload.get("company_specific", False)) else 0.0
        novelty_score = _normalize_novelty(str(payload.get("novelty", "medium")))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        risk_flags = payload.get("risk_flags", []) or []
        extracted_rows.append(
            {
                "ticker": str(row["ticker"]).upper(),
                "date": pd.Timestamp(row["published_at"]).normalize(),
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
        )
    return pd.DataFrame(extracted_rows)


def build_news_features() -> pd.DataFrame:
    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

    base = pd.read_parquet(FEATURES_PATH)[["ticker", "date"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"])
    news_rows = load_news_jsonl_files()
    if not news_rows:
        raise FileNotFoundError("No raw news JSONL files found. Run src/news_ingest.py first.")

    model_dir = _load_fingpt_model_dir()
    inferred = _infer_fingpt_event_rows(model_dir, news_rows)
    if inferred.empty:
        raise RuntimeError("FinGPT event extraction produced no rows.")

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
        {"source": "fingpt_model_inference", "rows": len(aggregated)},
    )

    result = base.merge(aggregated, on=["ticker", "date"], how="left").fillna(0.0)
    NEWS_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(NEWS_FEATURES_PATH, index=False)

    metadata = load_json(N1_PRODUCTION_DIR / "metadata.json")
    n1_metadata = {
        "model_name": "YOLO-WALLSTREET-n1",
        "model_version": metadata.get("model_version"),
        "artifact_path": metadata.get("artifact_path"),
        "type": "fingpt_structured_feature_extractor",
        "mac_inference_supported": False,
        "gpu_training_required": True,
    }
    save_json(N1_PRODUCTION_DIR / "metadata.json", n1_metadata)
    logging.info("Saved news features to %s with %s rows", NEWS_FEATURES_PATH, len(result))
    return result


def main() -> None:
    setup_logging()
    build_news_features()


if __name__ == "__main__":
    main()
