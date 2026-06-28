# YOLO-WALLSTREET

YOLO-WALLSTREET is a research and paper-trading stock prediction system built to train on NVIDIA GPUs and serve inference on a Mac.

It predicts **future returns**, then converts them into a projected close:

`expected_close = current_close * (1 + predicted_return)`

The intended question is:

`What will AAPL close at next Friday?`

The intended answer is:

- predicted return over the target horizon
- expected close from that return
- bull/bear range
- confidence
- main drivers
- risk flags

## Safety

YOLO-WALLSTREET is for research and paper trading only.

- No real-money trading logic
- No broker integration
- No secrets required for the local baseline

## Architecture

The active architecture is:

- `YOLO-WALLSTREET-t1`
  - `Chronos` price prior for time-series forecasting
  - current repo also keeps CUDA `XGBoost` training utilities for numeric price modeling
- `YOLO-WALLSTREET-n1`
  - `FinGPT`-style structured event extractor
  - input: real ingested news/articles
  - output: structured JSON features, not raw price predictions
- `YOLO-WALLSTREET-ensemble`
  - CUDA `XGBoost` baseline predictor
  - combines engineered market features, Chronos prior features, and FinGPT event features
- `YOLO-WALLSTREET-adjuster`
  - `Gemma` post-prediction adjuster
  - input: baseline predicted return, volatility/risk context, and fresh Bright Data news
  - output: bounded `adjustment_bps` applied to the baseline return

The final prediction flow is:

`ticker + market features + Chronos prior + FinGPT event features -> XGBoost baseline predicted_return -> Gemma news adjustment -> adjusted predicted_return -> expected_close`

## Strict Mode

This repo now runs in strict mode.

- no seeded example news
- no heuristic LLM fallback in live prediction
- no `t1` fallback when ensemble is missing
- no placeholder news-feature fallback in live prediction

If Chronos features, FinGPT artifacts, the XGBoost ensemble, or the Gemma adjuster artifacts are missing, the relevant commands fail loudly.

## Data Sources

### Market Data

Source:

- `yfinance`

Used for:

- OHLCV history
- labels
- benchmark context
- realized forward outcomes

Key files:

- [src/download_prices.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/download_prices.py)
- [src/build_features.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_features.py)
- [data/raw/prices/ohlcv_3y.parquet](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/raw/prices/ohlcv_3y.parquet)
- [data/processed/features.parquet](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/features.parquet)

### News / Event Data

Raw news is now populated by a real ingestion script.

Source path:

- [data/raw/news](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/raw/news)

Ingestion script:

- [src/news_ingest.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/news_ingest.py)

Current ingestion behavior:

- fetches public finance/news results from Google News RSS per ticker
- optionally routes requests through a Bright Data proxy if `BRIGHTDATA_PROXY_URL` or `YOLO_WALLSTREET_PROXY_URL` is set
- fetches article pages
- extracts text content
- dedupes and writes JSONL files

Raw news schema per line:

```json
{
  "ticker": "NVDA",
  "published_at": "2026-05-29T13:00:00+00:00",
  "title": "NVIDIA highlights continued AI data center demand",
  "body": "Article text...",
  "source": "Reuters",
  "url": "https://example.com/article"
}
```

## Model Training Data

### `t1 / Chronos`

Source:

- ticker-level market history from `yfinance`

Code:

- [src/train_t1_chronos.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_t1_chronos.py)

Output:

- [data/processed/chronos_features.parquet](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/chronos_features.parquet)

### `n1 / FinGPT`

Source:

- real ingested articles from [data/raw/news](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/raw/news)
- aligned market outcomes from [data/processed/features.parquet](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/features.parquet)

Training-data builder:

- [src/build_fingpt_training_data.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_fingpt_training_data.py)

Training dataset:

- [data/processed/n1_fingpt_train.jsonl](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/n1_fingpt_train.jsonl)

Important:

- FinGPT is not trained on raw OHLCV text
- it is trained on event text plus aligned market context and realized future-return labels

FinGPT training entrypoint:

- [src/train_n1_fingpt.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_n1_fingpt.py)

Structured event-feature generation:

- [src/build_news_features.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_news_features.py)

### `adjuster / Gemma`

Source:

- real market states from [data/processed/features.parquet](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/features.parquet)
- real raw news articles ingested from Bright Data into [data/raw/news](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/raw/news)
- built news features from [data/processed/news_features.parquet](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/news_features.parquet)
- baseline ensemble predictions produced by [src/train_ensemble.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_ensemble.py)

Training-data builder:

- [src/build_adjuster_training_data.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_adjuster_training_data.py)

Training dataset:

- [data/processed/adjuster_gemma_train.jsonl](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/processed/adjuster_gemma_train.jsonl)

Adjuster training entrypoint:

- [src/train_adjuster_gemma.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_adjuster_gemma.py)

Live adjuster inference:

- [src/adjuster.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/adjuster.py)

### `ensemble / XGBoost baseline`

Source:

- engineered price/regime features
- Chronos prior features
- FinGPT event features

Training entrypoint:

- [src/train_ensemble.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_ensemble.py)

## GPU Recommendation

For NVIDIA Brev:

- preferred: `A100 40GB`
- acceptable bring-up: `A10G 24GB`

If you want fewer VRAM constraints for Chronos + FinGPT + Gemma LoRA work, start with `A100 40GB`.

## NVIDIA GPU Training

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

Full strict training flow:

```bash
python3 src/download_prices.py
python3 src/build_features.py
python3 src/news_ingest.py --days-back 30
python3 src/train_t1_chronos.py --base-model amazon/chronos-2
python3 src/build_fingpt_training_data.py
python3 src/train_n1_fingpt.py --destination production --base-model FinGPT/fingpt-forecaster
python3 src/build_news_features.py
python3 src/train_t1_gpu.py --destination production
python3 src/train_ensemble.py --destination production
python3 src/build_adjuster_training_data.py
python3 src/train_adjuster_gemma.py --destination production --base-model google/gemma-3-4b-it
python3 src/init_db.py
python3 src/predict_ensemble.py --ticker AAPL --horizon 5d --log
```

Or run:

```bash
bash scripts/run_gpu_training.sh
```

## Real News Ingestion

Populate [data/raw/news](/Users/kaushiksivakumar/workspace/yolo-wallstreet/data/raw/news) with:

```bash
python src/news_ingest.py --days-back 7
```

Use a Bright Data proxy if you have one:

```bash
export BRIGHTDATA_PROXY_URL="http://USER:PASS@HOST:PORT"
python src/news_ingest.py --days-back 7
```

Use Bright Data API mode with the generic request endpoint:

```bash
export BRIGHTDATA_API_TOKEN="YOUR_API_KEY"
export BRIGHTDATA_REQUEST_ENDPOINT="https://api.brightdata.com/request"
export BRIGHTDATA_SERP_ZONE="serp_api1"
python src/news_ingest.py --days-back 7 --mode brightdata_api
```

You can also set `BRIGHTDATA_ZONE` instead of `BRIGHTDATA_SERP_ZONE` if you prefer one variable for the search zone.

## Mac Inference

Mac inference is supported for the numeric prediction service, but it assumes trained artifacts already exist and are copied from a GPU-trained environment.

From the GPU box, export a Mac bundle:

```bash
bash scripts/export_for_mac.sh
```

On the Mac:

```bash
cd yolo-wallstreet-mac
bash scripts/run_mac_inference.sh
```

Sample requests:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","horizon":"5d","log":true}'
```

Pre-setup requirements:

- training is NVIDIA-first
- inference on Mac should use prebuilt artifacts exported from the GPU machine
- `features.parquet`, `chronos_features.parquet`, and `news_features.parquet` must be present
- the Mac helper disables the Gemma adjuster by default via `YOLO_WALLSTREET_DISABLE_ADJUSTER=1`
- large local LLM inference on Mac is not the default path here

## Daily / Weekly Jobs

Daily after close:

```bash
bash scripts/run_daily_after_close.sh
```

This now does:

- refresh prices
- rebuild features
- ingest latest news
- rebuild FinGPT event features
- score mature predictions
- generate fresh paper predictions

Weekly retrain:

```bash
bash scripts/run_weekly_retrain.sh
```

This now does:

- refresh prices
- rebuild features
- ingest recent news
- rebuild FinGPT training data
- retrain FinGPT
- rebuild event features
- update outcomes
- retrain/evaluate/promote ensemble candidates
- rebuild adjuster training data from the new baseline ensemble
- retrain Gemma adjuster

## API

Run the API locally:

```bash
uvicorn src.api:app --reload --port 8000
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","horizon":"5d","log":true}'
```

## Main Files

- [src/news_ingest.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/news_ingest.py)
- [src/build_fingpt_training_data.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_fingpt_training_data.py)
- [src/build_news_features.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_news_features.py)
- [src/build_adjuster_training_data.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/build_adjuster_training_data.py)
- [src/train_t1_chronos.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_t1_chronos.py)
- [src/train_n1_fingpt.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_n1_fingpt.py)
- [src/train_adjuster_gemma.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_adjuster_gemma.py)
- [src/train_ensemble.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/train_ensemble.py)
- [src/predict_ensemble.py](/Users/kaushiksivakumar/workspace/yolo-wallstreet/src/predict_ensemble.py)
