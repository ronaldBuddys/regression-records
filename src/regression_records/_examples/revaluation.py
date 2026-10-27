"""A Black-Scholes book revaluation, for the gallery page ``13_pricing_pipeline``."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from regression_records import record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/revaluation, or a fresh temporary directory.
STORE = example_store("revaluation")

DAYS_PER_YEAR = 365.0  # ACT/365
NOISE = 0.0  # stands in for a BLAS or numpy upgrade

# erfc, not 1 + erf: the latter cancels to 0 in the left tail, deep out of the money.
_normal_cdf = np.vectorize(lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0)))


def black_scholes(spot, strike, vol, rate, tau, is_call):
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
MARKETS = [
    {"spot": 100.0, "vol": 0.2, "rate": 0.03},
    {"spot": 95.0, "vol": 0.25, "rate": 0.03},
]


def record_scenarios() -> list[pl.DataFrame]:
    """Value ``BOOK`` in each market scenario, and return the valuations."""
    return [value_book(BOOK, **market) for market in MARKETS]
