from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


PROJECT_ROOT = Path(os.getenv("YOLO_WALLSTREET_ROOT", Path(__file__).resolve().parents[1]))
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PRICES_DIR = RAW_DIR / "prices"
RAW_NEWS_DIR = RAW_DIR / "news"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_MODELS_DIR = MODELS_DIR / "production"
CANDIDATES_DIR = MODELS_DIR / "candidates"
ARCHIVED_MODELS_DIR = MODELS_DIR / "archived"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPORTS_DIR = PROJECT_ROOT / "exports"
DB_PATH = DATA_DIR / "yolo_wallstreet.db"
UNIVERSE_PATH = CONFIG_DIR / "universe.csv"
SETTINGS_PATH = CONFIG_DIR / "settings.yaml"

OHLCV_PATH = RAW_PRICES_DIR / "ohlcv_3y.parquet"
FAILED_TICKERS_PATH = RAW_PRICES_DIR / "failed_tickers.json"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
NEWS_FEATURES_PATH = PROCESSED_DIR / "news_features.parquet"
EXAMPLE_NEWS_PATH = RAW_NEWS_DIR / "example_news.jsonl"
N1_GEMMA_TRAIN_PATH = PROCESSED_DIR / "n1_gemma_train.jsonl"

T1_PRODUCTION_DIR = PRODUCTION_MODELS_DIR / "t1"
N1_PRODUCTION_DIR = PRODUCTION_MODELS_DIR / "n1"
ENSEMBLE_PRODUCTION_DIR = PRODUCTION_MODELS_DIR / "ensemble"

BENCHMARK_TICKERS = {"QQQ", "SPY", "XLK", "SMH", "ARKK"}
DEFAULT_HORIZON = "5d"

T1_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "volume_z20",
    "volume_ratio_20",
    "close_vs_ma10",
    "close_vs_ma20",
    "close_vs_ma50",
    "close_vs_ma200",
    "high_low_range",
    "close_open_return",
    "close_vs_20d_high",
    "close_vs_20d_low",
    "qqq_ret_1d",
    "qqq_ret_5d",
    "qqq_ret_20d",
    "spy_ret_1d",
    "spy_ret_5d",
    "spy_ret_20d",
    "stock_minus_qqq_ret_5d",
    "stock_minus_spy_ret_5d",
]

NEWS_FEATURE_COLUMNS = [
    "news_count",
    "avg_sentiment_stub",
    "max_positive_sentiment_stub",
    "max_negative_sentiment_stub",
    "earnings_count",
    "guidance_count",
    "ai_count",
    "lawsuit_count",
    "analyst_count",
    "regulatory_count",
]


def ensure_project_dirs() -> None:
    for path in [
        CONFIG_DIR,
        RAW_PRICES_DIR,
        RAW_NEWS_DIR,
        PROCESSED_DIR,
        PREDICTIONS_DIR,
        T1_PRODUCTION_DIR,
        N1_PRODUCTION_DIR,
        ENSEMBLE_PRODUCTION_DIR,
        CANDIDATES_DIR / "t1",
        CANDIDATES_DIR / "t1_gpu",
        CANDIDATES_DIR / "n1",
        CANDIDATES_DIR / "n1_gemma_lora",
        CANDIDATES_DIR / "ensemble",
        ARCHIVED_MODELS_DIR,
        REPORTS_DIR,
        EXPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
