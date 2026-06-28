# YOLO-WALLSTREET

YOLO-WALLSTREET is a research and paper-trading stock prediction system built around two models:

- `YOLO-WALLSTREET-t1`: a time-series / price model trained on OHLCV and market-regime features.
- `YOLO-WALLSTREET-n1`: a news model that converts headlines and articles into structured features.
- `YOLO-WALLSTREET-ensemble`: the final prediction layer that combines `t1` and `n1`.

The system predicts future returns, not magical exact close prices. It answers questions like:

`What will AAPL close at next Friday?`

internally as:

`ticker + as_of_date + target_horizon + features + news -> predicted_return -> expected_close`

where:

`expected_close = current_close * (1 + predicted_return)`

## Architecture

### YOLO-WALLSTREET-t1

- Baseline: `LightGBM` regressor on daily engineered features.
- Optional GPU path: PyTorch MLP baseline on the same tabular feature set.
- Future path: transformer time-series feature generation.

### YOLO-WALLSTREET-n1

- MVP: stub news feature extractor backed by JSONL news files.
- Output: structured daily features, not raw price predictions.
- Optional GPU path: Gemma LoRA training for richer feature extraction.

### Ensemble

- Trains on all `t1` features plus `n1` daily news features.
- Production inference is CPU-safe and Mac-friendly.

## Why Returns, Not Exact Prices

Predicting exact closes directly is fragile and misleading. YOLO-WALLSTREET predicts a future return for a horizon such as `5d`, then converts that return into an expected close using the current close. This keeps the modeling target coherent and allows confidence bands, bull/bear cases, and evaluation against realized returns.

## NVIDIA GPU Training Path

- `requirements-gpu.txt` adds optional NVIDIA-focused dependencies.
- `src/train_t1_gpu.py` trains a CUDA-first PyTorch tabular model.
- `src/train_n1_gemma_lora.py` is the optional Gemma LoRA training entrypoint.
- GPU scripts detect CUDA and exit with a useful message when it is unavailable.

## Mac Inference Path

Mac inference does not require CUDA.

- Production `t1` and ensemble models are stored as lightweight `joblib` artifacts.
- `n1` inference for MVP uses precomputed or stub news features.
- `src/device.py` detects `cuda`, `mps`, or `cpu`, but inference always works on CPU.

For future `n1` Mac deployment, use one of:

1. Precompute news features on a GPU machine and sync Parquet outputs to the Mac.
2. Export a quantized model later to `GGUF` or `MLX`.
3. Use hosted inference for the news model.

## Full Local Baseline

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_prices.py
python src/build_features.py
python src/build_news_features.py
python src/train_t1_lgbm.py
python src/train_ensemble.py
python src/init_db.py
python src/predict_ensemble.py --ticker AAPL --horizon 5d --log
uvicorn src.api:app --reload --port 8000
```

## GPU Training Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt
python src/download_prices.py
python src/build_features.py
python src/build_news_features.py
python src/train_t1_lgbm.py
python src/train_t1_gpu.py --allow-cpu false
python src/train_n1_gemma_lora.py
```

## Daily Self-Improvement Loop

```bash
bash scripts/run_daily_after_close.sh
```

This refreshes prices, rebuilds features, scores mature predictions, and generates new paper predictions for the universe.

## Weekly Retraining / Promotion Loop

```bash
bash scripts/run_weekly_retrain.sh
```

This retrains candidate models, evaluates them against production, backtests the candidate, and promotes only when the candidate passes the configured promotion rules.

## API Usage

Run the API locally:

```bash
uvicorn src.api:app --reload --port 8000
```

Sample prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","horizon":"5d","log":true}'
```

## Export For Mac

```bash
bash scripts/export_for_mac.sh
bash scripts/run_mac_inference.sh
```

The export bundle contains production models, config, source code, requirements, and a Mac inference README. It intentionally excludes large raw datasets.

## Safety

YOLO-WALLSTREET is for research and paper trading only.

- No real-money trading logic is included.
- No broker integration is included.
- No paid APIs or secrets are required for the MVP.
