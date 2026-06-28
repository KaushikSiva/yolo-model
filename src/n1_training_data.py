from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

from src.config import N1_GEMMA_TRAIN_PATH, ensure_project_dirs
from src.news_ingest import load_news_jsonl_files


TOPIC_KEYWORDS = {
    "earnings_count": {"earnings", "results"},
    "guidance_count": {"guidance", "outlook"},
    "ai_count": {"ai", "artificial intelligence", "gpu", "model"},
    "lawsuit_count": {"lawsuit", "litigation", "sued"},
    "analyst_count": {"analyst", "rating", "price target", "downgrade", "upgrade"},
    "regulatory_count": {"regulatory", "regulation", "antitrust", "investigation", "sec"},
}
POSITIVE_WORDS = {"beat", "strong", "growth", "upbeat", "surge", "win", "record", "expand", "positive"}
NEGATIVE_WORDS = {"miss", "weak", "cut", "drop", "decline", "lawsuit", "risk", "negative", "delay"}


def sentiment_stub(text: str) -> float:
    lowered = text.lower()
    positive_hits = sum(1 for word in POSITIVE_WORDS if word in lowered)
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in lowered)
    total = positive_hits + negative_hits
    if total == 0:
        return 0.0
    return float((positive_hits - negative_hits) / total)


def infer_catalyst_type(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in TOPIC_KEYWORDS["earnings_count"]):
        return "earnings"
    if any(word in lowered for word in TOPIC_KEYWORDS["guidance_count"]):
        return "guidance"
    if any(word in lowered for word in TOPIC_KEYWORDS["ai_count"]):
        return "AI_demand"
    if any(word in lowered for word in TOPIC_KEYWORDS["lawsuit_count"]):
        return "lawsuit"
    if any(word in lowered for word in TOPIC_KEYWORDS["regulatory_count"]):
        return "regulatory"
    if any(word in lowered for word in TOPIC_KEYWORDS["analyst_count"]):
        return "analyst_change"
    if "launch" in lowered or "product" in lowered:
        return "product_launch"
    return "general"


def infer_risk_flags(text: str, sentiment: float) -> list[str]:
    lowered = text.lower()
    risk_flags: list[str] = []
    if sentiment < -0.25:
        risk_flags.append("negative_sentiment")
    if "lawsuit" in lowered or "litigation" in lowered:
        risk_flags.append("lawsuit")
    if "regulatory" in lowered or "sec" in lowered or "antitrust" in lowered:
        risk_flags.append("regulatory")
    if "guidance" in lowered and ("cut" in lowered or "lower" in lowered):
        risk_flags.append("guidance_cut")
    if "valuation" in lowered:
        risk_flags.append("valuation")
    return risk_flags


def infer_likely_horizon(text: str) -> str:
    lowered = text.lower()
    if "today" in lowered or "tomorrow" in lowered or "this week" in lowered:
        return "1-5_trading_days"
    if "quarter" in lowered or "guidance" in lowered or "earnings" in lowered:
        return "5-20_trading_days"
    return "1-20_trading_days"


def build_weak_supervision_dataset(output_path: Path | None = None) -> Path | None:
    ensure_project_dirs()
    output_path = output_path or N1_GEMMA_TRAIN_PATH
    raw_rows = load_news_jsonl_files()
    if not raw_rows:
        return None

    training_rows = []
    for row in raw_rows:
        ticker = row.get("ticker", "UNKNOWN")
        published_at = row.get("published_at", "")
        title = row.get("title", "")
        body = row.get("body", "")
        text = f"{title}\n{body}".strip()
        sentiment = sentiment_stub(text)
        sentiment_label = "positive" if sentiment > 0.15 else "negative" if sentiment < -0.15 else "neutral"
        payload = {
            "sentiment": sentiment_label,
            "sentiment_score": round(abs(sentiment), 4),
            "catalyst_type": infer_catalyst_type(text),
            "company_specific": True,
            "risk_flags": infer_risk_flags(text, sentiment),
            "novelty": "medium",
            "likely_horizon": infer_likely_horizon(text),
            "confidence": 0.58,
        }
        training_rows.append(
            {
                "instruction": "Extract stock-relevant trading features as valid JSON.",
                "input": f"Ticker: {ticker}\nDate: {published_at[:10]}\nHeadline: {title}\nArticle: {body}",
                "output": json.dumps(payload, sort_keys=True),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in training_rows:
            handle.write(json.dumps(row) + "\n")
    return output_path
