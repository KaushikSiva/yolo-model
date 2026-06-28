from __future__ import annotations

import numpy as np
import pandas as pd

from src.build_features import add_group_features


def test_future_ret_5d_label_is_correct() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    closes = np.arange(100, 120, dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": np.full(len(dates), 1_000_000),
            "category": "mega_cap_tech",
        }
    )

    featured = add_group_features(frame)
    row = featured.iloc[0]
    expected = (closes[5] / closes[0]) - 1.0
    assert row["future_ret_5d"] == expected
