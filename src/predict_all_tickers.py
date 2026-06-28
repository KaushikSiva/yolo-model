from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

from src.config import DEFAULT_HORIZON, PREDICTIONS_DIR, ensure_project_dirs
from src.predict_ensemble import predict_for_ticker
from src.universe import get_non_benchmark_tickers
from src.utils import setup_logging, today_iso


def predict_all(horizon: str = DEFAULT_HORIZON, should_log: bool = True) -> list[dict]:
    ensure_project_dirs()
    outputs: list[dict] = []
    for ticker in get_non_benchmark_tickers():
        try:
            outputs.append(predict_for_ticker(ticker, horizon=horizon, should_log=should_log))
        except Exception as exc:
            outputs.append({"ticker": ticker, "error": str(exc)})

    output_path = PREDICTIONS_DIR / f"predictions_{today_iso().replace('-', '')}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row) + "\n")
    return outputs


def main() -> None:
    setup_logging()
    results = predict_all()
    print(f"Generated {len(results)} predictions.")


if __name__ == "__main__":
    main()
