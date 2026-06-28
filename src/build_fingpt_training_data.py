from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd

from src.config import FEATURES_PATH, N1_FINGPT_TRAIN_PATH, ensure_project_dirs
from src.news_ingest import load_news_jsonl_files
from src.utils import save_json, setup_logging


TOPIC_KEYWORDS = {
    "earnings": {"earnings", "results"},
    "guidance": {"guidance", "outlook"},
    "regulatory": {"regulatory", "regulation", "antitrust", "investigation", "sec"},
    "lawsuit": {"lawsuit", "litigation", "sued"},
    "analyst": {"analyst", "rating", "price target", "downgrade", "upgrade"},
    "ai_demand": {"ai", "artificial intelligence", "gpu", "model"},
}


def _lookup_feature_row(features: pd.DataFrame, ticker: str, published_at: pd.Timestamp) -> pd.Series | None:
    rows = features.loc[(features["ticker"] == ticker) & (features["date"] <= published_at.normalize())].sort_values("date")
    if rows.empty:
        return None
    return rows.iloc[-1]


def _label_catalyst(text: str) -> str:
    lowered = text.lower()
    for label, keywords in TOPIC_KEYWORDS.items():
        if any(word in lowered for word in keywords):
            return label
    return "general_news"


def _label_sentiment_from_outcome(feature_row: pd.Series | None) -> tuple[str, float]:
    if feature_row is None:
        return "neutral", 0.0
    future_ret_5d = float(feature_row.get("future_ret_5d", 0.0) or 0.0)
    score = max(-1.0, min(1.0, future_ret_5d / 0.08))
    if future_ret_5d > 0.02:
        return "positive", round(score, 4)
    if future_ret_5d < -0.02:
        return "negative", round(score, 4)
    return "neutral", round(score, 4)


def _label_horizon(feature_row: pd.Series | None) -> str:
    if feature_row is None:
        return "1_5_trading_days"
    spans = {
        "1_5_trading_days": max(
            abs(float(feature_row.get("future_ret_1d", 0.0) or 0.0)),
            abs(float(feature_row.get("future_ret_5d", 0.0) or 0.0)),
        ),
        "5_20_trading_days": max(
            abs(float(feature_row.get("future_ret_10d", 0.0) or 0.0)),
            abs(float(feature_row.get("future_ret_20d", 0.0) or 0.0)),
        ),
    }
    return max(spans, key=spans.get)


def _risk_flags(text: str, feature_row: pd.Series | None) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    if any(word in lowered for word in {"lawsuit", "litigation", "probe", "investigation"}):
        flags.append("legal_or_regulatory")
    if any(word in lowered for word in {"valuation", "multiple"}):
        flags.append("valuation")
    if feature_row is not None and float(feature_row.get("volatility_20d", 0.0) or 0.0) > 0.05:
        flags.append("high_volatility")
    if feature_row is not None and int(feature_row.get("rapid_move", 0) or 0) == 1:
        flags.append("rapid_move")
    return sorted(set(flags))


def build_fingpt_training_data() -> dict:
    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])
    news_rows = load_news_jsonl_files()
    if not news_rows:
        raise FileNotFoundError("No raw news JSONL files found. Run src/news_ingest.py first.")

    train_rows: list[dict] = []
    for row in news_rows:
        ticker = str(row["ticker"]).upper()
        published_at = pd.Timestamp(row["published_at"])
        text = f"{row.get('title', '')}\n{row.get('body', '')}".strip()
        feature_row = _lookup_feature_row(features, ticker, published_at)
        sentiment, sentiment_score = _label_sentiment_from_outcome(feature_row)
        likely_horizon = _label_horizon(feature_row)
        catalyst_type = _label_catalyst(text)
        risk_flags = _risk_flags(text, feature_row)
        confidence = 0.55
        if feature_row is not None:
            confidence += min(0.25, abs(float(feature_row.get("future_ret_5d", 0.0) or 0.0)) / 0.2)
        company_specific = ticker in text.upper()

        market_context = {}
        if feature_row is not None:
            market_context = {
                "ret_1d": round(float(feature_row.get("ret_1d", 0.0) or 0.0), 4),
                "ret_5d": round(float(feature_row.get("ret_5d", 0.0) or 0.0), 4),
                "volatility_20d": round(float(feature_row.get("volatility_20d", 0.0) or 0.0), 4),
                "future_ret_1d": round(float(feature_row.get("future_ret_1d", 0.0) or 0.0), 4),
                "future_ret_5d": round(float(feature_row.get("future_ret_5d", 0.0) or 0.0), 4),
                "future_ret_20d": round(float(feature_row.get("future_ret_20d", 0.0) or 0.0), 4),
            }

        output_payload = {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "catalyst_type": catalyst_type,
            "company_specific": bool(company_specific),
            "risk_flags": risk_flags,
            "novelty": "high" if any(word in text.lower() for word in {"guidance", "beat", "miss", "investigation"}) else "medium",
            "likely_horizon": likely_horizon,
            "confidence": round(min(confidence, 0.9), 4),
        }

        train_rows.append(
            {
                "instruction": "Extract stock-relevant trading features as valid JSON.",
                "input": (
                    f"Ticker: {ticker}\n"
                    f"Date: {published_at.date().isoformat()}\n"
                    f"Headline: {row.get('title', '')}\n"
                    f"Article: {row.get('body', '')}\n"
                    f"MarketContext: {json.dumps(market_context, sort_keys=True)}"
                ),
                "output": json.dumps(output_payload, sort_keys=True),
            }
        )

    N1_FINGPT_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with N1_FINGPT_TRAIN_PATH.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "training_rows": len(train_rows),
        "output_jsonl": str(N1_FINGPT_TRAIN_PATH),
        "source": "real_news_plus_market_outcomes",
    }
    save_json(N1_FINGPT_TRAIN_PATH.with_suffix(".metadata.json"), summary)
    return summary


def main() -> None:
    setup_logging()
    summary = build_fingpt_training_data()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
