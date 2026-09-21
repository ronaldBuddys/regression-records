"""The pricing code example 12 records: a Black-Scholes kernel applied to a book.

Conventions: spot and strike in currency, ``vol`` annualised, ``rate`` continuously
compounded, and expiry converted to a year fraction with ``DAYS_PER_YEAR``. Example 12
changes ``DAYS_PER_YEAR`` and ``NOISE`` to simulate code changes, and restores them.

``record_scenarios`` is the run that records: the book valued under two market
scenarios. Example 12 calls it in its own process and again in a fresh one.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "12_pricing_pipeline"

DAYS_PER_YEAR = 365.0  # ACT/365
NOISE = 0.0  # stands in for a BLAS or numpy upgrade

_normal_cdf = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def black_scholes(
    spot: float,
    strike: np.ndarray,
    vol: float,
    rate: float,
    tau: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Return European option prices; every argument broadcasts, ``tau`` is in years."""
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    call = spot * _normal_cdf(d1) - strike * np.exp(-rate * tau) * _normal_cdf(d2)
    put = call - spot + strike * np.exp(-rate * tau)  # put-call parity
    return np.where(is_call, call, put) * (1.0 + NOISE)


@record_func(STORE, tags={"suite": "revaluation"})
def value_book(
    book: pl.DataFrame, spot: float, vol: float, rate: float
) -> pl.DataFrame:
    """Return ``book`` with a per-contract ``price`` and a position ``value``."""
    tau = book["days_to_expiry"].to_numpy() / DAYS_PER_YEAR
    price = black_scholes(
        spot, book["strike"].to_numpy(), vol, rate, tau, book["is_call"].to_numpy()
    )
    return book.with_columns(
        price=pl.Series(price), value=pl.Series(price) * pl.col("quantity")
    )


BOOK = pl.DataFrame(
    {
        "trade_id": ["T1", "T2", "T3", "T4"],
        "strike": [90.0, 100.0, 100.0, 110.0],
        "days_to_expiry": [30, 90, 90, 180],
        "is_call": [True, True, False, False],
        "quantity": [10, -5, 20, 15],
    }
)


def record_scenarios() -> pl.DataFrame:
    """Value ``BOOK`` in two market scenarios, and return the first valuation."""
    valuation = value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
    value_book(BOOK, spot=95.0, vol=0.25, rate=0.03)
    return valuation
