from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from src.build_training_dataset import build_training_dataset
from src.config import DEFAULT_HORIZON, ENSEMBLE_PRODUCTION_DIR, N1_PRODUCTION_DIR, PLANNER_PRODUCTION_DIR, T1_CHRONOS_PRODUCTION_DIR, T1_PRODUCTION_DIR
from src.db import create_tables, get_engine, predictions_table
from src.evaluate_candidate import evaluate_candidate_model
from src.init_db import main as init_db_main
from src.predict_all_tickers import predict_all
from src.predict_ensemble import predict_for_ticker
from src.promote_model import promote_latest
from src.retrain_candidate import retrain_candidate_model
from src.update_outcomes import update_outcomes


app = FastAPI(title="YOLO-WALLSTREET")


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    horizon: str = DEFAULT_HORIZON
    log: bool = True


class CandidateRequest(BaseModel):
    model: str = Field(default="ensemble", pattern="^(t1|ensemble)$")


@app.on_event("startup")
def on_startup() -> None:
    create_tables(get_engine())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:
    payload = {}
    for name, path in {
        "t1": T1_PRODUCTION_DIR,
        "t1_chronos": T1_CHRONOS_PRODUCTION_DIR,
        "n1": N1_PRODUCTION_DIR,
        "ensemble": ENSEMBLE_PRODUCTION_DIR,
        "planner": PLANNER_PRODUCTION_DIR,
    }.items():
        metadata_path = path / "metadata.json"
        payload[name] = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    return payload


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return predict_for_ticker(request.ticker.upper(), request.horizon, request.log)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/score-outcomes")
def score_outcomes() -> dict:
    return {"updated": update_outcomes()}


@app.post("/predict-all")
def predict_all_endpoint() -> dict:
    results = predict_all()
    return {"count": len(results), "results": results[:5]}


@app.post("/retrain-candidate")
def retrain_candidate(request: CandidateRequest) -> dict:
    return retrain_candidate_model(request.model)


@app.post("/evaluate-candidate")
def evaluate_candidate(request: CandidateRequest) -> dict:
    return evaluate_candidate_model(request.model)


@app.post("/promote-latest")
def promote_candidate(request: CandidateRequest) -> dict:
    return promote_latest(request.model)


@app.get("/predictions/{ticker}")
def recent_predictions(ticker: str) -> dict:
    engine = get_engine()
    with engine.begin() as connection:
        rows = connection.execute(
            select(predictions_table)
            .where(predictions_table.c.ticker == ticker.upper())
            .order_by(desc(predictions_table.c.created_at))
            .limit(25)
        ).mappings().all()
    return {"ticker": ticker.upper(), "predictions": [dict(row) for row in rows]}
