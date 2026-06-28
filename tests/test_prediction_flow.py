from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

from src import config as config_module
from src import db as db_module
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
    n1_dir = tmp_path / "models" / "production" / "n1"
    ensemble_dir = tmp_path / "models" / "production" / "ensemble"
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

    x = pd.DataFrame([{column: feature_row[column] for column in config_module.T1_FEATURE_COLUMNS}])
    y = [0.02]
    t1_model = DummyRegressor(strategy="constant", constant=0.02).fit(x, y)
    t1_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(t1_model, t1_dir / "model.joblib")
    _write_metadata(
        t1_dir,
        {
            "model_name": "YOLO-WALLSTREET-t1",
            "model_version": "YOLO-WALLSTREET-t1-vtest",
            "feature_columns": config_module.T1_FEATURE_COLUMNS,
        },
    )

    ensemble_columns = config_module.T1_FEATURE_COLUMNS + config_module.NEWS_FEATURE_COLUMNS
    ensemble_x = pd.DataFrame([{column: feature_row.get(column, 0.0) for column in ensemble_columns}])
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
    monkeypatch.setattr(predict_t1_module, "T1_PRODUCTION_DIR", t1_dir)
    monkeypatch.setattr(predict_n1_module, "NEWS_FEATURES_PATH", news_path)
    monkeypatch.setattr(predict_n1_module, "N1_PRODUCTION_DIR", n1_dir)
    monkeypatch.setattr(predict_ensemble_module, "FEATURES_PATH", features_path)
    monkeypatch.setattr(predict_ensemble_module, "NEWS_FEATURES_PATH", news_path)
    monkeypatch.setattr(predict_ensemble_module, "T1_PRODUCTION_DIR", t1_dir)
    monkeypatch.setattr(predict_ensemble_module, "ENSEMBLE_PRODUCTION_DIR", ensemble_dir)
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
    }
    assert required_keys.issubset(prediction.keys())

    prediction_id = log_prediction(prediction, features_snapshot={"stub": True}, db_path=str(db_path))
    engine = get_engine(str(db_path))
    with engine.begin() as connection:
        rows = list(connection.exec_driver_sql("SELECT prediction_id FROM predictions"))
    assert any(row[0] == prediction_id for row in rows)
