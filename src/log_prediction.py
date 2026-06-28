from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sqlalchemy import insert

from src.db import create_tables, get_engine, predictions_table
from src.utils import json_dumps, make_id, utc_now_iso


def log_prediction(prediction: dict, features_snapshot: dict | None = None, db_path: str | None = None) -> str:
    create_tables(get_engine(db_path))
    prediction_id = make_id("pred")
    row = {
        "prediction_id": prediction_id,
        "created_at": utc_now_iso(),
        "ticker": prediction["ticker"],
        "as_of_date": prediction["as_of_date"],
        "target_horizon": prediction["target_horizon"],
        "target_date": prediction["target_date"],
        "current_close": prediction["current_close"],
        "predicted_close": prediction["expected_close"],
        "predicted_return": prediction["predicted_return"],
        "bear_case": prediction["bear_case"],
        "bull_case": prediction["bull_case"],
        "confidence": prediction["confidence"],
        "confidence_score": prediction["confidence_score"],
        "t1_model_version": prediction["model_versions"].get("t1"),
        "n1_model_version": prediction["model_versions"].get("n1"),
        "ensemble_model_version": prediction["model_versions"].get("ensemble"),
        "features_json": json_dumps(features_snapshot or {}),
    }
    engine = get_engine(db_path)
    with engine.begin() as connection:
        connection.execute(insert(predictions_table).values(**row))
    return prediction_id
