from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

from src import config as config_module
from src import db as db_module
from src import planner as planner_module
from src import predict_ensemble as predict_ensemble_module
from src import predict_n1 as predict_n1_module
from src import predict_t1 as predict_t1_module
from src.db import create_tables, get_engine
from src.log_prediction import log_prediction


def _write_metadata(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_output_and_logging(tmp_path, monkeypatch) -> None:
    features_path = tmp_path / "features.parquet"
    news_path = tmp_path / "news_features.parquet"
    t1_dir = tmp_path / "models" / "production" / "t1"
    t1_chronos_dir = tmp_path / "models" / "production" / "t1_chronos"
    chronos_path = tmp_path / "chronos_features.parquet"
    n1_dir = tmp_path / "models" / "production" / "n1"
    ensemble_dir = tmp_path / "models" / "production" / "ensemble"
    planner_dir = tmp_path / "models" / "production" / "planner"
    db_path = tmp_path / "test.db"

    feature_row = {
        "ticker": "AAPL",
        "date": "2026-06-20",
        "close": 200.0,
        "rapid_move": 0,
        "volatility_20d": 0.02,
    }
    for idx, column in enumerate(config_module.T1_FEATURE_COLUMNS):
        feature_row[column] = 0.01 * (idx + 1)
    pd.DataFrame([feature_row]).to_parquet(features_path, index=False)

    news_row = {"ticker": "AAPL", "date": "2026-06-20"}
    for column in config_module.NEWS_FEATURE_COLUMNS:
        news_row[column] = 0.0
    pd.DataFrame([news_row]).to_parquet(news_path, index=False)

    chronos_row = {"ticker": "AAPL", "date": "2026-06-20"}
    for column in config_module.CHRONOS_FEATURE_COLUMNS:
        chronos_row[column] = 0.01
    pd.DataFrame([chronos_row]).to_parquet(chronos_path, index=False)
    t1_chronos_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        t1_chronos_dir,
        {
            "model_name": "YOLO-WALLSTREET-t1",
            "model_version": "YOLO-WALLSTREET-t1-chronos-vtest",
            "feature_columns": config_module.CHRONOS_FEATURE_COLUMNS,
            "implementation_backend": "chronos_2",
        },
    )

    ensemble_columns = config_module.T1_FEATURE_COLUMNS + config_module.CHRONOS_FEATURE_COLUMNS + config_module.NEWS_FEATURE_COLUMNS
    ensemble_x = pd.DataFrame([{column: feature_row.get(column, 0.0) for column in ensemble_columns}])
    for column in config_module.CHRONOS_FEATURE_COLUMNS:
        ensemble_x[column] = chronos_row[column]
    y = [0.02]
    ensemble_model = DummyRegressor(strategy="constant", constant=0.02).fit(ensemble_x, y)
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble_model, ensemble_dir / "model.joblib")
    _write_metadata(
        ensemble_dir,
        {
            "model_name": "YOLO-WALLSTREET-ensemble",
            "model_version": "YOLO-WALLSTREET-ensemble-vtest",
            "feature_columns": ensemble_columns,
        },
    )

    _write_metadata(
        n1_dir,
        {
            "model_name": "YOLO-WALLSTREET-n1",
            "model_version": "n1_stub_v1",
        },
    )

    monkeypatch.setattr(predict_t1_module, "FEATURES_PATH", features_path)
    monkeypatch.setattr(predict_t1_module, "CHRONOS_FEATURES_PATH", chronos_path)
    monkeypatch.setattr(predict_t1_module, "T1_CHRONOS_PRODUCTION_DIR", t1_chronos_dir)
    monkeypatch.setattr(predict_n1_module, "NEWS_FEATURES_PATH", news_path)
    monkeypatch.setattr(predict_n1_module, "N1_PRODUCTION_DIR", n1_dir)
    monkeypatch.setattr(predict_ensemble_module, "ENSEMBLE_PRODUCTION_DIR", ensemble_dir)
    monkeypatch.setattr(planner_module, "PLANNER_PRODUCTION_DIR", planner_dir)
    monkeypatch.setattr(
        predict_ensemble_module,
        "load_price_features",
        lambda: pd.read_parquet(features_path).assign(date=lambda df: pd.to_datetime(df["date"])),
    )
    monkeypatch.setattr(
        predict_ensemble_module,
        "load_chronos_features",
        lambda: pd.read_parquet(chronos_path).assign(date=lambda df: pd.to_datetime(df["date"])),
    )
    monkeypatch.setattr(
        predict_t1_module,
        "load_chronos_features",
        lambda: pd.read_parquet(chronos_path).assign(date=lambda df: pd.to_datetime(df["date"])),
    )
    monkeypatch.setattr(
        predict_ensemble_module,
        "plan_retrieval",
        lambda **_: {
            "planner_model_version": "YOLO-WALLSTREET-planner-vtest",
            "planner_backend": "gemma_local",
            "should_retrieve": False,
            "urgency": "low",
            "urgency_score": 0.1,
            "target_horizon": "5d",
            "ticker": "AAPL",
            "current_close": 200.0,
            "triggers": [],
            "suggested_sources": [],
            "query_terms": [],
            "notes": "test",
        },
    )
    monkeypatch.setattr(db_module, "DB_PATH", db_path)

    create_tables(get_engine(str(db_path)))
    prediction = predict_ensemble_module.predict_for_ticker("AAPL", horizon="5d", should_log=False)

    required_keys = {
        "ticker",
        "as_of_date",
        "target_horizon",
        "target_date",
        "current_close",
        "expected_close",
        "predicted_return",
        "bear_case",
        "bull_case",
        "confidence",
        "confidence_score",
        "model_versions",
        "main_drivers",
        "risk_flags",
        "planner",
        "sources_used",
    }
    assert required_keys.issubset(prediction.keys())
    assert "planner" in prediction["model_versions"]

    prediction_id = log_prediction(prediction, features_snapshot={"stub": True}, db_path=str(db_path))
    engine = get_engine(str(db_path))
    with engine.begin() as connection:
        rows = list(connection.exec_driver_sql("SELECT prediction_id, planner_json FROM predictions"))
    assert any(row[0] == prediction_id and row[1] for row in rows)
