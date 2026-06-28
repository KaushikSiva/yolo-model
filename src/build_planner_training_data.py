from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from collections import defaultdict

from src.config import DEFAULT_HORIZON, FEATURES_PATH, NEWS_FEATURES_PATH, PLANNER_GEMMA_TRAIN_PATH, ensure_project_dirs
from src.feature_store import load_news_features, load_price_features
from src.news_ingest import load_news_jsonl_files
from src.utils import setup_logging


def _source_map() -> dict[tuple[str, str], set[str]]:
    source_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in load_news_jsonl_files():
        ticker = str(row["ticker"]).upper()
        date = str(row["published_at"])[:10]
        source_lookup[(ticker, date)].add(str(row.get("source", "")).lower())
    return source_lookup


def _plan_from_real_state(row: dict, sources_seen: set[str], horizon: str) -> dict:
    triggers: list[str] = []
    score = 0.1
    if int(row.get("rapid_move", 0) or 0) == 1:
        triggers.append("rapid_move")
        score += 0.35
    if float(row.get("volatility_20d", 0.0) or 0.0) > 0.05:
        triggers.append("high_volatility")
        score += 0.2
    if float(row.get("news_count", 0.0) or 0.0) == 0.0:
        triggers.append("missing_news")
        score += 0.2
    if abs(float(row.get("future_ret_1d", 0.0) or 0.0)) > 0.03 or abs(float(row.get("future_ret_5d", 0.0) or 0.0)) > 0.05:
        triggers.append("large_realized_move")
        score += 0.15
    if float(row.get("materiality_score", 0.0) or 0.0) > 0.6:
        triggers.append("material_news_context")
        score += 0.1
    if float(row.get("earnings_count", 0.0) or 0.0) > 0 or float(row.get("guidance_count", 0.0) or 0.0) > 0:
        triggers.append("earnings_or_guidance")
        score += 0.1

    should_retrieve = score >= 0.35
    urgency = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"
    suggested_sources = ["public_finance_news", "company_ir", "sec_filings"]
    if "reuters" in sources_seen or "yahoo" in sources_seen:
        suggested_sources.insert(0, "follow_existing_public_news")
    if "sec" in sources_seen:
        suggested_sources.insert(0, "sec_filings")
    if "investor" in " ".join(sources_seen):
        suggested_sources.insert(0, "company_ir")
    if "earnings_or_guidance" in triggers:
        suggested_sources.insert(0, "earnings_transcript")

    deduped_sources = []
    for source in suggested_sources:
        if source not in deduped_sources:
            deduped_sources.append(source)

    ticker = str(row["ticker"]).upper()
    return {
        "should_retrieve": should_retrieve,
        "urgency": urgency,
        "urgency_score": round(min(score, 0.95), 4),
        "triggers": triggers,
        "suggested_sources": deduped_sources if should_retrieve else [],
        "query_terms": [f"{ticker} stock news", f"{ticker} sec filing", f"{ticker} investor relations"] if should_retrieve else [],
        "notes": "Generated from real market state and real ingested news availability.",
        "target_horizon": horizon,
    }


def build_planner_training_data(horizon: str = DEFAULT_HORIZON) -> dict:
    ensure_project_dirs()
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")
    if not NEWS_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing news features file: {NEWS_FEATURES_PATH}")

    prices = load_price_features()
    news = load_news_features()
    merged = prices.merge(news, on=["ticker", "date"], how="left").fillna(0.0).sort_values(["ticker", "date"])
    source_lookup = _source_map()

    rows: list[dict] = []
    sample = merged.tail(min(len(merged), 4000))
    for _, row in sample.iterrows():
        row_dict = row.to_dict()
        key = (str(row_dict["ticker"]).upper(), row_dict["date"].date().isoformat())
        plan = _plan_from_real_state(row_dict, source_lookup.get(key, set()), horizon)
        rows.append(
            {
                "instruction": "Generate a stock-news retrieval plan as valid JSON.",
                "input": (
                    f"Ticker: {row_dict['ticker']}\n"
                    f"Date: {row_dict['date'].date().isoformat()}\n"
                    f"Horizon: {horizon}\n"
                    f"ret_1d: {float(row_dict.get('ret_1d', 0.0) or 0.0):.4f}\n"
                    f"ret_5d: {float(row_dict.get('ret_5d', 0.0) or 0.0):.4f}\n"
                    f"volatility_20d: {float(row_dict.get('volatility_20d', 0.0) or 0.0):.4f}\n"
                    f"rapid_move: {int(row_dict.get('rapid_move', 0) or 0)}\n"
                    f"news_count: {float(row_dict.get('news_count', 0.0) or 0.0):.0f}\n"
                    f"materiality_score: {float(row_dict.get('materiality_score', 0.0) or 0.0):.4f}\n"
                    f"sources_seen: {sorted(source_lookup.get(key, set()))}"
                ),
                "output": json.dumps(
                    {
                        "should_retrieve": plan["should_retrieve"],
                        "urgency": plan["urgency"],
                        "urgency_score": plan["urgency_score"],
                        "triggers": plan["triggers"],
                        "suggested_sources": plan["suggested_sources"],
                        "query_terms": plan["query_terms"],
                        "notes": plan["notes"],
                    },
                    sort_keys=True,
                ),
            }
        )

    with PLANNER_GEMMA_TRAIN_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "training_rows": len(rows),
        "output_jsonl": str(PLANNER_GEMMA_TRAIN_PATH),
        "source": "real_market_state_plus_real_ingested_news",
    }


def main() -> None:
    setup_logging()
    summary = build_planner_training_data()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
