"""
Pricing pipeline : record, change, replay
=========================================

The earlier examples take one feature at a time. This one puts them together the way a
real suite would: a small pricing pipeline - a numpy kernel and a polars valuation
table - recorded once, then replayed after each of three changes: numerical noise that
must pass, a real regression that must fail, and a deliberate change that is accepted
by re-recording.

It needs polars as well as numpy.
"""

import inspect
import sys
from contextlib import contextmanager
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import polars as pl
from _shared import revaluation
from _shared.demo import fresh_store, record_paths, show_cases, show_result

from regression_records import RecordStore, ToleranceConfig, replay

STORE = fresh_store(revaluation.STORE.name)

# %%
# The code under test
# -------------------
#
# A Black-Scholes kernel and the function that applies it to a book, in
# ``examples/_shared/revaluation.py``. Conventions: spot and strike in currency,
# ``vol`` annualised, ``rate`` continuously compounded, and expiry converted to a year
# fraction with ``DAYS_PER_YEAR`` - which is the thing a later section changes.

print(inspect.getsource(revaluation.value_book))

# %%
# Recording
# ---------
#
# Run the pipeline on the scenarios worth pinning. The book is a DataFrame *argument*,
# so it goes to a parquet side file, and the valuation to another. Two market scenarios
# are two records.

BOOK = pl.DataFrame(
    {
        "trade_id": ["T1", "T2", "T3", "T4"],
        "strike": [90.0, 100.0, 100.0, 110.0],
        "days_to_expiry": [30, 90, 90, 180],
        "is_call": [True, True, False, False],
        "quantity": [10, -5, 20, 15],
    }
)

valuation = revaluation.value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
revaluation.value_book(BOOK, spot=95.0, vol=0.25, rate=0.03)
print(valuation)
print("records:", len(record_paths(STORE)))
print("side files:", sorted(p.suffix for p in STORE.rglob("data/*")))

# %%
# The tolerance
# -------------
#
# Tolerance is decided at replay, not at recording. Prices are floats and will not
# reproduce bit for bit across machines, so they get a relative tolerance. Quantities
# and days are integers, which always compare exactly; identifiers are strings, which
# have no tolerance at all.
#
# A position in a deep out-of-the-money option is worth almost nothing, and a relative
# tolerance shrinks with it. The ``path_overrides`` entry gives the ``value`` column an
# absolute floor of a millionth of a currency unit; it inherits ``rtol=1e-9`` from the
# config above it, since an override changes only the fields it is given.

TOLERANCE = ToleranceConfig(
    rtol=1e-9, path_overrides={"$.output['value']": ToleranceConfig(atol=1e-6)}
)

report = replay(STORE, tolerance=TOLERANCE)
print(report.summary())


# %%
# Each of the changes below is made by setting a module-level constant of
# ``revaluation`` - standing in for an edit to the file - and undone afterwards whatever
# happens.


@contextmanager
def changed(**settings):
    """Temporarily set module-level constants of ``_shared.revaluation``."""
    saved = {name: getattr(revaluation, name) for name in settings}
    for name, value in settings.items():
        setattr(revaluation, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(revaluation, name, value)


# %%
# Numerical noise
# ---------------
#
# Upgrade numpy, move to another CPU, and the last few bits of a float change. That is
# not a regression, and ``rtol=1e-9`` absorbs it: this replay perturbs every price by
# one part in 10^12 and still passes.

with changed(NOISE=1e-12):
    report = replay(STORE, tolerance=TOLERANCE)
print(report.summary())

# %%
# A real regression
# -----------------
#
# Someone "tidies" the day count to ACT/360. Nothing crashes and the prices look
# plausible. Replay names the columns that moved and leaves alone the ones that did
# not - ``quantity`` and ``days_to_expiry`` are untouched, so they are not in the
# report.

with changed(DAYS_PER_YEAR=360.0):
    report = replay(STORE, tolerance=TOLERANCE)
show_cases(report)
print()
show_result(report.failures[0].comparison)

# %%
# ``code_drift`` is false: ``value_book``'s own source is unchanged, so only the values
# could have caught this.
#
# Accepting a deliberate change
# -----------------------------
#
# Suppose ACT/360 was the right call after all. The old records now describe behaviour
# you no longer want, so they are removed - by hand, since records are never deleted
# for you - and the scenarios recorded again. ``prune`` then clears side files no record
# refers to any more.
#
# In a real repository the store is committed, so ``git diff`` on it is the review: the
# old records out, the new ones in.

store = RecordStore(STORE)
with changed(DAYS_PER_YEAR=360.0):
    for path in store.iter_record_paths():
        if store.load_record(path).identity.qualname == "value_book":
            path.unlink()
    revaluation.value_book(BOOK, spot=100.0, vol=0.2, rate=0.03)
    revaluation.value_book(BOOK, spot=95.0, vol=0.25, rate=0.03)
    print("pruned:", [p.name for p in store.prune(dry_run=False)])
    print(replay(STORE, tolerance=TOLERANCE).summary())

# %%
# Removing them is the deliberate act. A stale record left in the store keeps failing
# replay, which is accurate - it and the code disagree - until someone decides which of
# the two is right.
#
# In a test file
# --------------
#
# In a suite the replay is one line. ``replay_tests`` (example 14) makes each record its
# own pytest test, so a failure names the scenario that moved:
#
# .. code-block:: python
#
#    # tests/test_regressions.py
#    from regression_records import ToleranceConfig, replay_tests
#
#    test_regressions = replay_tests(
#        "tests/regression_data",
#        tolerance_overrides={
#            "mylib.revaluation:*": ToleranceConfig(
#                rtol=1e-9,
#                path_overrides={"$.output['value']": ToleranceConfig(atol=1e-6)},
#            )
#        },
#    )
#
#
# Commit ``tests/regression_data/`` alongside it, and a pull request that moves a price
# fails with the columns that moved.
