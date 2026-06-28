from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np
import pandas as pd

from src.config import CHRONOS_FEATURE_COLUMNS, DEFAULT_HORIZON, ENSEMBLE_PRODUCTION_DIR, T1_FEATURE_COLUMNS
from src.feature_store import load_chronos_features, load_price_features
from src.log_prediction import log_prediction
from src.modeling import load_model_bundle
from src.planner import plan_retrieval
from src.predict_n1 import predict_n1
from src.predict_t1 import predict_t1
from src.utils import business_day_offset, clamp, confidence_label, summarize_feature_drivers


def _load_feature_row(ticker: str) -> pd.Series:
    df = load_price_features()
    rows = df.loc[df["ticker"] == ticker].dropna(subset=["close"]).sort_values("date")
    if rows.empty:
        raise ValueError(f"No features found for ticker {ticker}")
    return rows.iloc[-1]


def _load_news_payload(ticker: str, as_of_date: str) -> dict:
    return predict_n1(ticker, as_of_date)


def _load_chronos_row(ticker: str, as_of_date: str) -> dict[str, float]:
    chronos = load_chronos_features()
    if chronos.empty:
        return {column: 0.0 for column in CHRONOS_FEATURE_COLUMNS}
    rows = chronos.loc[(chronos["ticker"] == ticker) & (chronos["date"] <= pd.Timestamp(as_of_date))].sort_values("date")
    if rows.empty:
        return {column: 0.0 for column in CHRONOS_FEATURE_COLUMNS}
    row = rows.iloc[-1]
    return {column: float(row.get(column, 0.0) or 0.0) for column in CHRONOS_FEATURE_COLUMNS}


def _compute_confidence(predicted_return: float, volatility_20d: float, missing_news: bool, t1_pred: float, ensemble_pred: float) -> tuple[str, float]:
    score = 0.62
    if volatility_20d > 0.05:
        score -= 0.18
    elif volatility_20d > 0.03:
        score -= 0.08
    if missing_news:
        score -= 0.08
    if abs(predicted_return) < 0.005:
        score -= 0.08
    if np.sign(t1_pred) == np.sign(ensemble_pred) and abs(t1_pred - ensemble_pred) < 0.01:
        score += 0.08
    score = clamp(score, 0.2, 0.95)
    return confidence_label(score), round(score, 4)


def predict_for_ticker(ticker: str, horizon: str = DEFAULT_HORIZON, should_log: bool = False) -> dict:
    ticker = ticker.upper()
    t1_payload = predict_t1(ticker)
    row = _load_feature_row(ticker)
    current_close = float(row["close"])
    as_of_date = row["date"].date().isoformat()
    news_payload = _load_news_payload(ticker, as_of_date)
    news_features = news_payload["news_features"]
    chronos_features = _load_chronos_row(ticker, as_of_date)
    missing_news = False

    ensemble_model, ensemble_metadata = load_model_bundle(ENSEMBLE_PRODUCTION_DIR)
    feature_columns = ensemble_metadata["feature_columns"]
    feature_values = {column: float(row.get(column, 0.0) or 0.0) for column in T1_FEATURE_COLUMNS}
    feature_values.update(chronos_features)
    feature_values.update(news_features)
    feature_frame = pd.DataFrame([feature_values])
    predicted_return = float(ensemble_model.predict(feature_frame[feature_columns])[0])
    ensemble_version = ensemble_metadata["model_version"]
    importances = getattr(ensemble_model, "feature_importances_", None)
    driver_row = pd.Series(feature_values)

    target_date = business_day_offset(as_of_date, horizon)
    expected_close = current_close * (1.0 + predicted_return)
    volatility = float(row.get("volatility_20d", 0.02) or 0.02)
    spread = max(volatility, 0.015)
    bear_case = current_close * (1.0 + predicted_return - spread)
    bull_case = current_close * (1.0 + predicted_return + spread)
    confidence, confidence_score = _compute_confidence(
        predicted_return=predicted_return,
        volatility_20d=volatility,
        missing_news=missing_news,
        t1_pred=float(t1_payload["t1_predicted_return"]),
        ensemble_pred=predicted_return,
    )
    planner = plan_retrieval(
        ticker=ticker,
        feature_row=row,
        news_features=news_features,
        t1_payload=t1_payload,
        horizon=horizon,
        current_close=current_close,
    )
    if planner["should_retrieve"]:
        confidence_score = round(clamp(confidence_score - 0.05, 0.2, 0.95), 4)
        confidence = confidence_label(confidence_score)

    risk_flags = []
    if missing_news:
        risk_flags.append("missing_news_features")
    if volatility > 0.05:
        risk_flags.append("high_volatility")
    if int(row.get("rapid_move", 0) or 0) == 1:
        risk_flags.append("recent_rapid_move")
    if abs(predicted_return) > 0.08:
        risk_flags.append("large_predicted_move")
    if planner["should_retrieve"]:
        risk_flags.append("planner_requests_live_retrieval")

    result = {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "target_horizon": horizon,
        "target_date": target_date,
        "current_close": round(current_close, 4),
        "expected_close": round(expected_close, 4),
        "predicted_return": round(predicted_return, 6),
        "bear_case": round(bear_case, 4),
        "bull_case": round(bull_case, 4),
        "confidence": confidence,
        "confidence_score": confidence_score,
        "model_versions": {
            "t1": t1_payload["t1_model_version"],
            "n1": news_payload["n1_model_version"],
            "ensemble": ensemble_version,
            "planner": planner["planner_model_version"],
        },
        "main_drivers": summarize_feature_drivers(driver_row, feature_columns, importances=importances, limit=5),
        "risk_flags": risk_flags,
        "planner": planner,
        "sources_used": planner["suggested_sources"],
    }

    if should_log:
        prediction_id = log_prediction(
            result,
            features_snapshot={"t1": t1_payload, "chronos": chronos_features, "n1": news_features, "planner": planner},
        )
        result["prediction_id"] = prediction_id

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    print(json.dumps(predict_for_ticker(args.ticker, args.horizon, args.log), indent=2))


if __name__ == "__main__":
    main()
