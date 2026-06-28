from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EXAMPLE_NEWS_PATH, FEATURES_PATH, NEWS_FEATURES_PATH, N1_PRODUCTION_DIR, RAW_NEWS_DIR, ensure_project_dirs
from src.utils import save_json, setup_logging


POSITIVE_WORDS = {"beat", "strong", "growth", "upbeat", "surge", "win", "record", "expand", "positive"}
NEGATIVE_WORDS = {"miss", "weak", "cut", "drop", "decline", "lawsuit", "risk", "negative", "delay"}
TOPIC_KEYWORDS = {
    "earnings_count": ["earnings", "results"],
    "guidance_count": ["guidance", "outlook"],
    "ai_count": ["ai", "artificial intelligence", "gpu", "model"],
    "lawsuit_count": ["lawsuit", "litigation", "sued"],
    "analyst_count": ["analyst", "rating", "price target", "downgrade", "upgrade"],
    "regulatory_count": ["regulatory", "regulation", "antitrust", "investigation", "sec"],
}


def load_news_jsonl_files() -> list[dict]:
    if not EXAMPLE_NEWS_PATH.exists():
        from src.news_ingest_stub import create_example_news_file

        create_example_news_file()

    rows: list[dict] = []
    for path in sorted(RAW_NEWS_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def sentiment_stub(text: str) -> float:
    lowered = text.lower()
    positive_hits = sum(1 for word in POSITIVE_WORDS if word in lowered)
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in lowered)
    total = positive_hits + negative_hits
    if total == 0:
        return 0.0
    return float((positive_hits - negative_hits) / total)


def build_news_features() -> pd.DataFrame:
    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

    base = pd.read_parquet(FEATURES_PATH)[["ticker", "date"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"])

    news_rows = load_news_jsonl_files()
    if not news_rows:
        result = base.copy()
        for column in [
            "news_count",
            "avg_sentiment_stub",
            "max_positive_sentiment_stub",
            "max_negative_sentiment_stub",
            "earnings_count",
            "guidance_count",
            "ai_count",
            "lawsuit_count",
            "analyst_count",
            "regulatory_count",
        ]:
            result[column] = 0.0
    else:
        news = pd.DataFrame(news_rows)
        news["published_at"] = pd.to_datetime(news["published_at"])
        news["date"] = news["published_at"].dt.normalize()
        news["text"] = news[["title", "body"]].fillna("").agg(" ".join, axis=1)
        news["sentiment"] = news["text"].map(sentiment_stub)
        news["positive_sentiment"] = news["sentiment"].clip(lower=0.0)
        news["negative_sentiment"] = news["sentiment"].clip(upper=0.0)

        for feature_name, keywords in TOPIC_KEYWORDS.items():
            news[feature_name] = news["text"].str.lower().apply(
                lambda text, words=keywords: int(any(word in text for word in words))
            )

        grouped = news.groupby(["ticker", "date"], as_index=False).agg(
            news_count=("text", "count"),
            avg_sentiment_stub=("sentiment", "mean"),
            max_positive_sentiment_stub=("positive_sentiment", "max"),
            max_negative_sentiment_stub=("negative_sentiment", "min"),
            earnings_count=("earnings_count", "sum"),
            guidance_count=("guidance_count", "sum"),
            ai_count=("ai_count", "sum"),
            lawsuit_count=("lawsuit_count", "sum"),
            analyst_count=("analyst_count", "sum"),
            regulatory_count=("regulatory_count", "sum"),
        )
        result = base.merge(grouped, on=["ticker", "date"], how="left").fillna(0.0)

    NEWS_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(NEWS_FEATURES_PATH, index=False)

    n1_metadata = {
        "model_name": "YOLO-WALLSTREET-n1",
        "model_version": "n1_stub_v1",
        "type": "stub_news_feature_extractor",
        "mac_inference_supported": True,
        "gpu_training_required": False,
    }
    save_json(N1_PRODUCTION_DIR / "metadata.json", n1_metadata)
    logging.info("Saved news features to %s with %s rows", NEWS_FEATURES_PATH, len(result))
    return result


def main() -> None:
    setup_logging()
    build_news_features()


if __name__ == "__main__":
    main()
