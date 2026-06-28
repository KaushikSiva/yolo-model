from __future__ import annotations

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

from src.config import DB_PATH, ensure_project_dirs


metadata = MetaData()

predictions_table = Table(
    "predictions",
    metadata,
    Column("prediction_id", String, primary_key=True),
    Column("created_at", String, nullable=False),
    Column("ticker", String, nullable=False),
    Column("as_of_date", String, nullable=False),
    Column("target_horizon", String, nullable=False),
    Column("target_date", String, nullable=False),
    Column("current_close", Float, nullable=False),
    Column("predicted_close", Float, nullable=False),
    Column("predicted_return", Float, nullable=False),
    Column("bear_case", Float, nullable=False),
    Column("bull_case", Float, nullable=False),
    Column("confidence", String, nullable=False),
    Column("confidence_score", Float, nullable=False),
    Column("t1_model_version", String),
    Column("n1_model_version", String),
    Column("ensemble_model_version", String),
    Column("features_json", Text),
    Column("actual_close", Float),
    Column("actual_return", Float),
    Column("absolute_error", Float),
    Column("return_error", Float),
    Column("direction_correct", Integer),
    Column("within_prediction_range", Integer),
    Column("scored_at", String),
)

model_runs_table = Table(
    "model_runs",
    metadata,
    Column("run_id", String, primary_key=True),
    Column("created_at", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("run_type", String, nullable=False),
    Column("metrics_json", Text),
    Column("promoted", Integer, nullable=False, default=0),
)

user_feedback_table = Table(
    "user_feedback",
    metadata,
    Column("feedback_id", String, primary_key=True),
    Column("prediction_id", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("rating", Integer),
    Column("feedback_text", Text),
)


def get_engine(db_path: str | None = None) -> Engine:
    ensure_project_dirs()
    target = db_path or DB_PATH
    return create_engine(f"sqlite:///{target}", future=True)


def create_tables(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    metadata.create_all(engine)
