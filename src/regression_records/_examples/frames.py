"""Order books as polars and pandas frames, for the gallery page ``11_dataframes``."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import polars as pl

from regression_records import record_func

# Where records go is configuration, set by whoever runs this code; never set here.
STORE = os.environ["REGRESSION_RECORDS_STORE"]

MIDS = [
    101.25,
    99.5,
    100.03,
    98.75,
    102.0,
    100.4,
    97.5,
    101.12,
    99.75,
    100.5,
    98.2,
    103.0,
]
SPREADS = [0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.5, 0.5]
TICK = None  # when set, every quote is rounded to a multiple of it


def book_data(n: int) -> dict[str, list | np.ndarray]:
    """Return ``n`` rows of symbols, quotes and volumes as columns."""
    mid, spread = np.array(MIDS[:n]), np.array(SPREADS[:n])
    bid, ask = mid - spread / 2, mid + spread / 2
    if TICK is not None:
        bid, ask = np.round(bid / TICK) * TICK, np.round(ask / TICK) * TICK
    return {
        "symbol": [f"S{i:02d}" for i in range(n)],
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "volume": np.arange(1, n + 1) * 100,
    }


@record_func(STORE)
def polars_book(n: int) -> pl.DataFrame:
    """Return a book of ``n`` rows as a polars frame."""
    return pl.DataFrame(book_data(n))


@record_func(STORE)
def polars_notional(book: pl.DataFrame, fx: float) -> pl.DataFrame:
    """Return ``book`` with a ``notional`` column, in the currency ``fx`` converts to."""
    return book.with_columns(notional=pl.col("mid") * pl.col("volume") * fx)


@record_func(STORE)
def polars_returns(prices: pl.Series) -> pl.Series:
    """Return the simple period returns of ``prices``, named ``"return"``."""
    return prices.pct_change().alias("return")


@record_func(STORE)
def pandas_book(n: int) -> pd.DataFrame:
    """Return a book of ``n`` rows as a pandas frame indexed by symbol."""
    return pd.DataFrame(book_data(n)).set_index("symbol")


@record_func(STORE)
def pandas_notional(book: pd.DataFrame, fx: float) -> pd.DataFrame:
    """Return ``book`` with a ``notional`` column, in the currency ``fx`` converts to."""
    return book.assign(notional=book["mid"] * book["volume"] * fx)


@record_func(STORE)
def pandas_rolling(values: list[float]) -> pd.Series:
    """Return the two-day rolling mean of ``values``, indexed by calendar day."""
    series = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))
    return series.rolling(2).mean()
