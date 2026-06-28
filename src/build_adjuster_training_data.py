from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from collections import defaultdict

import pandas as pd

from src.adjuster import LOOKBACK_DAYS, MAX_ADJUSTMENT_BPS
from src.config import ADJUSTER_GEMMA_TRAIN_PATH, DEFAULT_HORIZON, ENSEMBLE_PRODUCTION_DIR, ensure_project_dirs
from src.feature_store import load_training_frame
from src.modeling import load_model_bundle, prepare_model_frame
from src.news_ingest import load_news_jsonl_files
from src.utils import clamp, setup_logging


def _normalize_news_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _recent_news_map(lookback_days: int = LOOKBACK_DAYS, max_items: int = 3) -> dict[tuple[str, str], list[dict]]:
    rows = load_news_jsonl_files()
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_ticker[str(row["ticker"]).upper()].append(row)
    for ticker_rows in by_ticker.values():
        ticker_rows.sort(key=lambda item: _normalize_news_timestamp(str(item.get("published_at", ""))))

    lookup: dict[tuple[str, str], list[dict]] = {}
    for ticker, ticker_rows in by_ticker.items():
        for row in ticker_rows:
            published_day = _normalize_news_timestamp(row["published_at"])
            key = (ticker, published_day.date().isoformat())
            lookup.setdefault(key, [])

        all_dates = sorted({_normalize_news_timestamp(row["published_at"]) for row in ticker_rows})
        for date_value in all_dates:
            window_start = date_value - pd.Timedelta(days=lookback_days)
            items: list[dict] = []
            for row in reversed(ticker_rows):
                published_at = _normalize_news_timestamp(row["published_at"])
                if published_at > date_value or published_at < window_start:
                    continue
                items.append(
                    {
                        "title": str(row.get("title", "")).strip(),
                        "source": str(row.get("source", "")).strip(),
                        "body_excerpt": str(row.get("body", "")).strip()[:220],
                    }
                )
                if len(items) >= max_items:
                    break
            lookup[(ticker, date_value.date().isoformat())] = items
    return lookup


def _baseline_risk_flags(row: dict, baseline_predicted_return: float) -> list[str]:
    flags: list[str] = []
    if float(row.get("volatility_20d", 0.0) or 0.0) > 0.05:
        flags.append("high_volatility")
    if int(row.get("rapid_move", 0) or 0) == 1:
        flags.append("recent_rapid_move")
    if abs(baseline_predicted_return) > 0.08:
        flags.append("large_baseline_move")
    if float(row.get("risk_flag_count", 0.0) or 0.0) > 0:
        flags.append("structured_news_risk_flags")
    return flags


def _cited_signals(row: dict) -> list[str]:
    signals: list[str] = []
    if float(row.get("earnings_count", 0.0) or 0.0) > 0:
        signals.append("earnings_headlines")
    if float(row.get("guidance_count", 0.0) or 0.0) > 0:
        signals.append("guidance_change")
    if float(row.get("analyst_count", 0.0) or 0.0) > 0:
        signals.append("analyst_activity")
    if float(row.get("regulatory_count", 0.0) or 0.0) > 0:
        signals.append("regulatory_context")
    if float(row.get("materiality_score", 0.0) or 0.0) > 0.6:
        signals.append("high_materiality_news")
    if float(row.get("avg_sentiment_stub", 0.0) or 0.0) > 0.2:
        signals.append("positive_sentiment")
    if float(row.get("avg_sentiment_stub", 0.0) or 0.0) < -0.2:
        signals.append("negative_sentiment")
    return signals[:5]


def _target_adjustment_bps(row: dict, baseline_predicted_return: float) -> int:
    residual_bps = (float(row["future_ret_5d"]) - baseline_predicted_return) * 10_000.0
    news_weight = 0.0
    if float(row.get("news_count", 0.0) or 0.0) > 0:
        news_weight = clamp(
            0.25
            + 0.4 * float(row.get("materiality_score", 0.0) or 0.0)
            + 0.2 * float(row.get("event_confidence_score", 0.0) or 0.0)
            + 0.15 * min(float(row.get("news_count", 0.0) or 0.0) / 3.0, 1.0),
            0.0,
            1.0,
        )
    adjusted = residual_bps * news_weight
    return int(round(clamp(adjusted, -MAX_ADJUSTMENT_BPS, MAX_ADJUSTMENT_BPS)))


def build_adjuster_training_data(horizon: str = DEFAULT_HORIZON) -> dict:
    ensure_project_dirs()
    if not (ENSEMBLE_PRODUCTION_DIR / "metadata.json").exists():
        raise FileNotFoundError(
            f"Missing ensemble metadata: {ENSEMBLE_PRODUCTION_DIR / 'metadata.json'}. Train the XGBoost ensemble first."
        )

    model, metadata = load_model_bundle(ENSEMBLE_PRODUCTION_DIR)
    feature_columns = metadata["feature_columns"]
    merged = load_training_frame(include_news=True, include_chronos=True).sort_values(["ticker", "date"])
    frame = prepare_model_frame(merged, feature_columns, target_column="future_ret_5d")
    sample = frame.tail(min(len(frame), 4000)).copy()
    sample["baseline_predicted_return"] = model.predict(sample[feature_columns])
    news_lookup = _recent_news_map()

    rows: list[dict] = []
    for _, row in sample.iterrows():
        row_dict = row.to_dict()
        as_of_date = row_dict["date"].date().isoformat()
        ticker = str(row_dict["ticker"]).upper()
        recent_news = news_lookup.get((ticker, as_of_date), [])
        target_adjustment_bps = _target_adjustment_bps(row_dict, float(row_dict["baseline_predicted_return"]))
        baseline_risk_flags = _baseline_risk_flags(row_dict, float(row_dict["baseline_predicted_return"]))
        cited_signals = _cited_signals(row_dict)
        confidence = clamp(
            0.25
            + 0.35 * float(row_dict.get("materiality_score", 0.0) or 0.0)
            + 0.25 * float(row_dict.get("event_confidence_score", 0.0) or 0.0)
            + 0.15 * min(float(row_dict.get("news_count", 0.0) or 0.0) / 3.0, 1.0),
            0.1,
            0.95,
        )
        output_payload = {
            "adjustment_bps": target_adjustment_bps,
            "confidence": round(confidence, 4),
            "rationale": "Adjust the XGBoost baseline only when recent news plausibly changes near-term return expectations.",
            "cited_signals": cited_signals,
            "risk_flags": baseline_risk_flags,
        }
        rows.append(
            {
                "instruction": "Adjust a stock baseline forecast using fresh news context as valid JSON.",
                "input": (
                    f"Ticker: {ticker}\n"
                    f"Date: {as_of_date}\n"
                    f"Horizon: {horizon}\n"
                    f"Current close: {float(row_dict['close']):.4f}\n"
                    f"Baseline predicted_return: {float(row_dict['baseline_predicted_return']):.6f}\n"
                    f"Volatility_20d: {float(row_dict.get('volatility_20d', 0.0) or 0.0):.6f}\n"
                    f"Baseline risk flags: {baseline_risk_flags}\n"
                    f"Structured news features: "
                    f"{{'news_count': {float(row_dict.get('news_count', 0.0) or 0.0):.0f}, "
                    f"'avg_sentiment_stub': {float(row_dict.get('avg_sentiment_stub', 0.0) or 0.0):.4f}, "
                    f"'materiality_score': {float(row_dict.get('materiality_score', 0.0) or 0.0):.4f}, "
                    f"'event_confidence_score': {float(row_dict.get('event_confidence_score', 0.0) or 0.0):.4f}}}\n"
                    f"Recent news items: {recent_news}"
                ),
                "output": json.dumps(output_payload, sort_keys=True),
            }
        )

    with ADJUSTER_GEMMA_TRAIN_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "training_rows": len(rows),
        "output_jsonl": str(ADJUSTER_GEMMA_TRAIN_PATH),
        "source": "ensemble_baseline_plus_recent_raw_news",
    }


def main() -> None:
    setup_logging()
    summary = build_adjuster_training_data()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
