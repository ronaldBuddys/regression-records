"""
A realistic case: revaluing an option book
==========================================

The earlier examples take one feature at a time. This one puts them together the way a
real suite would: a small pricing pipeline - a numpy kernel, a polars valuation table,
and a stateful ledger - pinned by records, run one way on a laptop and another in CI,
surviving numerical noise, catching a real regression, and accepting a deliberate change.

It needs polars as well as numpy.
"""

import math
import os
import sys
import warnings
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import numpy as np
import polars as pl
from _shared.demo import fresh_store, record_paths, show_result

from regression_records import (
    OverwritePolicy,
    RegressionMismatchError,
    RunMode,
    ToleranceConfig,
    UnmatchedInvocationError,
    record_func,
    record_method,
)

STORE = fresh_store("12_pricing_pipeline")

# %%
# The code under test
# -------------------
#
# A Black-Scholes kernel and the function that applies it to a book. Conventions: spot
# and strike in currency, ``vol`` annualised, ``rate`` continuously compounded, and
# expiry converted to a year fraction with the day count below - which is the thing a
# later section changes.

DAYS_PER_YEAR = 365.0  # ACT/365
NOISE = 0.0  # stands in for a BLAS or numpy upgrade; see "Numerical noise" below

_normal_cdf = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))


def black_scholes(spot, strike, vol, rate, tau, is_call):
    """Price European options; every argument broadcasts, ``tau`` is in years."""
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    call = spot * _normal_cdf(d1) - strike * np.exp(-rate * tau) * _normal_cdf(d2)
    put = call - spot + strike * np.exp(-rate * tau)  # put-call parity
    return np.where(is_call, call, put) * (1.0 + NOISE)


# %%
# Choosing the mode from the environment
# --------------------------------------
#
# On a laptop the default ``RunMode.BOTH`` records new cases as you write them. In CI
# nothing may be recorded: a case with no history must fail, or the suite can pass by
# recording whatever the code does today. One line decides which, and every decoration
# shares it.
#
# Most CI systems set ``CI=true``, so ``os.environ.get("CI")`` is the usual switch. This
# script uses a variable of its own because it has to be able to record when it is
# itself run in CI, with nothing recorded beforehand.

MODE = (
    RunMode.VALIDATE_ONLY
    if os.environ.get("REVALUATION_VALIDATE_ONLY")
    else RunMode.BOTH
)

# %%
# Per-column tolerances
# ---------------------
#
# Prices are floats and will not reproduce bit for bit across machines, so they get a
# relative tolerance. Quantities are integers and identifiers are strings: they are
# exact, and saying so keeps the default ``rtol`` from forgiving a position that moved
# by one part in ten million. The paths name columns of the returned frame.

EXACT = ToleranceConfig(rtol=0.0, atol=0.0)
TOLERANCE = ToleranceConfig(
    rtol=1e-9,
    path_overrides={
        "$.output['quantity']": EXACT,
        "$.output['days_to_expiry']": EXACT,
    },
)


@record_func(STORE, mode=MODE, tolerance=TOLERANCE, tags={"suite": "revaluation"})
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

valuation = value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
valuation

# %%
# That call was recorded. The book is a DataFrame *argument*, so it went to a parquet
# side file, and the valuation went to another. Two market scenarios are two cases:

value_book(BOOK, spot=95.0, vol=0.25, rate=0.03)
print("records:", len(record_paths(STORE)))
print("side files:", sorted(p.suffix for p in STORE.rglob("data/*")))

# %%
# A method with state
# -------------------
#
# The ledger accumulates. ``record_method`` snapshots it before and after each call, so
# the test covers the mutation and not just the return value. ``_audit`` holds
# timestamps - different on every run for uninteresting reasons - so it is excluded.


class Ledger:
    def __init__(self):
        self.marks = {}
        self.total = 0.0
        self._audit = []

    @record_method(STORE, mode=MODE, tolerance=TOLERANCE, exclude_attrs=["_audit"])
    def mark(self, trade_id: str, value: float) -> float:
        """Record a trade's value and return the change in the book total."""
        import time

        change = value - self.marks.get(trade_id, 0.0)
        self.marks[trade_id] = value
        self.total += change
        self._audit.append(time.time())
        return change


def mark_book(frame: pl.DataFrame) -> Ledger:
    ledger = Ledger()
    for trade_id, value in frame.select("trade_id", "value").iter_rows():
        ledger.mark(trade_id, value)
    return ledger


print("book total:", round(mark_book(valuation).total, 4))

# %%
# Each ``mark`` call was matched by its *arguments* - trade and value - and the ledger's
# state rode along. Run it again and all four calls validate, state included:

mark_book(valuation)
print("passed:", Ledger.mark.last_result.passed)

# %%
# Numerical noise
# ---------------
#
# Upgrade numpy, move to another CPU, and the last few bits of a float change. That is
# not a regression, and ``rtol=1e-9`` absorbs it: the run below perturbs every price by
# one part in 10^12 and still passes.

NOISE = 1e-12
value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
print("passed:", value_book.last_result.passed)
NOISE = 0.0

# %%
# A real regression
# -----------------
#
# Someone "tidies" the day count to ACT/360. Nothing crashes and the prices look
# plausible. The failure names the columns that moved and leaves alone the ones that did
# not - ``quantity`` and ``days_to_expiry`` are untouched, so they are not in the report.

DAYS_PER_YEAR = 360.0

try:
    value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
except RegressionMismatchError as e:
    show_result(e.result)

# %%
# What CI would do with an unseen scenario
# ----------------------------------------
#
# Under ``VALIDATE_ONLY`` a scenario nobody recorded is an error, not a new baseline.
# (``tolerance`` and the identity are the same, so this decoration reads the same
# records.)

DAYS_PER_YEAR = 365.0
ci_value_book = record_func(STORE, mode="validate_only", tolerance=TOLERANCE)(
    value_book.__wrapped__
)

ci_value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
print("known scenario passed:", ci_value_book.last_result.passed)

try:
    ci_value_book(BOOK, spot=120.0, vol=0.2, rate=0.03)
except UnmatchedInvocationError as e:
    print("unseen scenario:", str(e)[:75], "...")

# %%
# Accepting a deliberate change
# -----------------------------
#
# Suppose ACT/360 was the right call after all. Re-run once with
# ``OverwritePolicy.IF_MISMATCH``: records that no longer match are replaced, and the diff
# stays on ``last_result`` so you can read what you accepted. In a real repository the
# store is committed, so ``git diff`` on it is the review.

DAYS_PER_YEAR = 360.0
accepting = record_func(
    STORE, tolerance=TOLERANCE, overwrite=OverwritePolicy.IF_MISMATCH
)(value_book.__wrapped__)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the drift warning is example 7's subject
    accepting(BOOK, spot=100.0, vol=0.2, rate=0.03)
print("accepted", len(accepting.last_result.diffs), "difference(s)")

value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
print("and the normal decoration now passes:", value_book.last_result.passed)

# %%
# In a test file
# --------------
#
# None of this needs a plugin. The decorated function is called from an ordinary test,
# and a mismatch is an ordinary exception that fails it:
#
# .. code-block:: python
#
#    # tests/test_revaluation.py
#    import pytest
#    from mylib.revaluation import value_book   # decorated as above
#
#    @pytest.mark.parametrize("spot, vol", [(100.0, 0.2), (95.0, 0.25)])
#    def test_revaluation_is_unchanged(book, spot, vol):
#        value_book(book, spot=spot, vol=vol, rate=0.03)
#
# Commit ``tests/regression_data/`` alongside it, set ``CI=1`` in the workflow, and a
# pull request that moves a price fails with the columns that moved.
