from __future__ import annotations

import pandas as pd

from src.build_fingpt_training_data import _lookup_feature_row, _normalize_news_timestamp


def test_normalize_news_timestamp_converts_tz_aware_to_naive_midnight() -> None:
    normalized = _normalize_news_timestamp("2026-06-26T15:52:03+00:00")

    assert normalized == pd.Timestamp("2026-06-26")
    assert normalized.tzinfo is None


def test_lookup_feature_row_handles_tz_aware_news_timestamp() -> None:
    features = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2026-06-25", "2026-06-26"]),
            "future_ret_5d": [0.01, 0.02],
        }
    )

    row = _lookup_feature_row(features, "AAPL", pd.Timestamp("2026-06-26T15:52:03+00:00"))

    assert row is not None
    assert float(row["future_ret_5d"]) == 0.02
