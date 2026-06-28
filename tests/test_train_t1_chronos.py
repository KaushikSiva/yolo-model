from __future__ import annotations

import pandas as pd

from src.train_t1_chronos import _chronos_context_df


def test_chronos_context_df_regularizes_trading_history_to_business_days() -> None:
    history = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": "2026-06-24", "close": 100.0},
            {"ticker": "AAPL", "date": "2026-06-25", "close": 101.0},
            {"ticker": "AAPL", "date": "2026-06-26", "close": 102.0},
            {"ticker": "AAPL", "date": "2026-06-29", "close": 103.0},
        ]
    )
    history["date"] = pd.to_datetime(history["date"])

    context = _chronos_context_df(history)

    assert context["id"].tolist() == ["AAPL", "AAPL", "AAPL", "AAPL"]
    assert context["target"].tolist() == [100.0, 101.0, 102.0, 103.0]
    assert pd.infer_freq(context["timestamp"]) == "B"
