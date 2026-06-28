from __future__ import annotations

import pandas as pd

from src.config import UNIVERSE_PATH


def load_universe() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_PATH)
    universe["is_benchmark"] = universe["is_benchmark"].astype(int)
    return universe


def get_tickers(include_benchmarks: bool = True) -> list[str]:
    universe = load_universe()
    if include_benchmarks:
        return universe["ticker"].tolist()
    return universe.loc[universe["is_benchmark"] == 0, "ticker"].tolist()


def get_non_benchmark_tickers() -> list[str]:
    return get_tickers(include_benchmarks=False)


def get_category_map() -> dict[str, str]:
    universe = load_universe()
    return dict(zip(universe["ticker"], universe["category"]))
