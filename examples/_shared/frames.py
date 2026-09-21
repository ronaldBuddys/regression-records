"""The functions example 10 records: dataframes and series as arguments and results.

Each exists in a polars and a pandas version. ``SPREAD`` shifts every price
``book_data`` generates; example 10 sets it to show column-level diffs, then restores
it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "10_dataframes"

SPREAD = 0.0  # added to every generated price


def book_data(n: int) -> dict[str, list | np.ndarray]:
    """Return ``n`` rows of symbols, prices and volumes as columns."""
    return {
        "symbol": [f"S{i:02d}" for i in range(n)],
        "price": np.linspace(10.0, 20.0, n) + SPREAD,
        "volume": np.arange(n) * 100,
    }


@record_func(STORE)
def polars_book(n: int) -> pl.DataFrame:
    """Return a book of ``n`` rows as a polars frame."""
    return pl.DataFrame(book_data(n))


@record_func(STORE)
def polars_notional(book: pl.DataFrame, fx: float) -> pl.DataFrame:
    """Return ``book`` with a ``notional`` column, in the currency ``fx`` converts to."""
    return book.with_columns(notional=pl.col("price") * pl.col("volume") * fx)


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
    return book.assign(notional=book["price"] * book["volume"] * fx)


@record_func(STORE)
def pandas_rolling(values: list[float]) -> pd.Series:
    """Return the two-day rolling mean of ``values``, indexed by calendar day."""
    series = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))
    return series.rolling(2).mean()
