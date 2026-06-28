from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime

from src.build_features import build_features
from src.build_news_features import build_news_features
from src.config import CANDIDATES_DIR
from src.train_ensemble import train_ensemble_model
from src.train_t1_lgbm import train_t1_model
from src.utils import setup_logging


def retrain_candidate_model(model_type: str) -> dict:
    build_features()
    build_news_features()
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = CANDIDATES_DIR / model_type / timestamp
    if model_type == "t1":
        return train_t1_model(output_dir=output_dir)
    if model_type == "ensemble":
        return train_ensemble_model(output_dir=output_dir)
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["t1", "ensemble"], required=True)
    args = parser.parse_args()
    setup_logging()
    metadata = retrain_candidate_model(args.model)
    print(metadata["model_version"])


if __name__ == "__main__":
    main()
