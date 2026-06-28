from __future__ import annotations

import numpy as np
import pandas as pd

from src.build_features import add_group_features


def test_feature_generation_on_synthetic_data() -> None:
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": np.linspace(100, 130, len(dates)),
            "high": np.linspace(101, 131, len(dates)),
            "low": np.linspace(99, 129, len(dates)),
            "close": np.linspace(100, 130, len(dates)),
            "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
            "category": "mega_cap_tech",
        }
    )

    featured = add_group_features(frame)
    latest = featured.iloc[-1]

    assert "ret_5d" in featured.columns
    assert "ma200" in featured.columns
    assert "future_ret_5d" in featured.columns
    assert not pd.isna(latest["ma200"])
    assert latest["volume_ratio_20"] > 0
