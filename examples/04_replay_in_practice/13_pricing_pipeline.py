"""
``replay``, ``replay_tests`` : a pricing pipeline, recorded, changed, replayed
==============================================================================

The earlier pages take one feature at a time. This one puts them together the way a
real suite would: a small pricing pipeline - a numpy kernel and a polars valuation
table - recorded once, then replayed after each of three changes: numerical noise that
must pass, a real regression that must fail, and a deliberate change that is accepted
by re-recording.

It needs polars as well as numpy.
"""

# %%
# Setup: where records go
# -----------------------
#
# The decorators are given no store: they read ``REGRESSION_RECORDS_STORE`` from the
# environment or a ``.env`` file. This page points it at a fresh temporary directory.

import os
import tempfile
from pathlib import Path

STORE = Path(tempfile.mkdtemp(prefix="regression-records-"))
os.environ["REGRESSION_RECORDS_STORE"] = str(STORE)

# %%
# Setup: the code under test
# --------------------------
#
# Replay imports a recorded function by module and name, so the code under test lives in
# a module, ``revaluation``, imported from ``regression_records._examples`` - support
# code that ships with the package so that a page can import real code rather than
# define it in a cell (:ref:`sphx_glr_auto_examples_01_basics_01_first_record.py` says
# more). In your project, this is your own package. Its source, printed below:
#
# A Black-Scholes kernel and the function that applies it to a book. Conventions: spot
# and strike in currency, ``vol`` annualised, ``rate`` continuously compounded, and
# expiry converted to a year fraction with ``DAYS_PER_YEAR`` - which is the thing a
# later section changes. ``NOISE`` stands in for a BLAS or numpy upgrade.
#
# ``record_scenarios`` is the run that records: one book valued under two markets.

import inspect
import subprocess
import sys
from contextlib import contextmanager

from regression_records._examples import revaluation

print(inspect.getsource(revaluation))

# %%
# Recording
# ---------
#
# Recording is off unless switched on; this page switches it on for the whole script.
# Then run the pipeline on the scenarios worth pinning. The book is a DataFrame
# *argument*, so it goes to a parquet side file, and the valuation to another. Two
# market scenarios are two records.

import numpy as np

from regression_records import RecordStore, ToleranceConfig, enable, recording, replay

enable()

valuations = revaluation.record_scenarios()
print(revaluation.BOOK)
print(valuations[0])
assert len(list(STORE.rglob("rec-*.json"))) == 2
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

RTOL = 1e-9
TOLERANCE = ToleranceConfig(
    rtol=RTOL, path_overrides={"$.output['value']": ToleranceConfig(atol=1e-6)}
)

report = replay(tolerance=TOLERANCE)
print(report.summary())
assert report.passed


# %%
# Each of the changes below is made by setting a module-level constant of
# ``revaluation`` - standing in for an edit to the file - and undone afterwards whatever
# happens. Under each change the page also values the book itself, with recording off,
# to plot what replay compared.


@contextmanager
def changed(**settings):
    """Temporarily set module-level constants of ``revaluation``."""
    saved = {name: getattr(revaluation, name) for name in settings}
    for name, value in settings.items():
        setattr(revaluation, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(revaluation, name, value)


def relative_price_change() -> np.ndarray:
    """Return each position's relative price change against the recorded valuations."""
    with recording(False):
        now = revaluation.record_scenarios()
    return np.concatenate(
        [
            np.abs(new["price"].to_numpy() / old["price"].to_numpy() - 1)
            for new, old in zip(now, valuations, strict=True)
        ]
    )


# %%
# Numerical noise
# ---------------
#
# Upgrade numpy, move to another CPU, and the last few bits of a float change. That is
# not a regression, and ``rtol=1e-9`` absorbs it: this replay perturbs every price by
# one part in 10^12 and still passes.

with changed(NOISE=1e-12):
    report = replay(tolerance=TOLERANCE)
    noise = relative_price_change()
print(report.summary())
assert report.passed

# %%
# A real regression
# -----------------
#
# Someone "tidies" the day count to ACT/360. Nothing crashes and the prices look
# plausible. Replay names the columns that moved and leaves alone the ones that did
# not - ``quantity`` and ``days_to_expiry`` are untouched, so they are not in the
# report.

with changed(DAYS_PER_YEAR=360.0):
    report = replay(tolerance=TOLERANCE)
    act360 = relative_price_change()
print(report.counts())
print(report.failures[0].comparison.summary())

paths = {d.path for case in report.failures for d in case.comparison.diffs}
assert report.counts()["differs"] == 2
assert paths == {"$.output['price']", "$.output['value']"}

# %%
# Side by side, the two changes are eight orders of magnitude apart, and the tolerance
# sits between them: every position moved under both, but only the regression moved
# beyond ``rtol``.

import matplotlib.pyplot as plt

assert noise.max() < RTOL < act360.min()

fig, ax = plt.subplots(figsize=(7.0, 4.0))
rng = np.random.default_rng(0)
for x, moves in enumerate([noise, act360]):
    beyond = moves > RTOL
    jitter = x + rng.uniform(-0.12, 0.12, moves.size)
    ax.scatter(
        jitter[~beyond],
        moves[~beyond],
        s=44,
        color="#0ca30c",
        zorder=3,
        label="within tolerance" if x == 0 else None,
    )
    ax.scatter(
        jitter[beyond],
        moves[beyond],
        s=44,
        color="#d03b3b",
        zorder=3,
        label="beyond tolerance" if x == 1 else None,
    )
ax.axhline(RTOL, color="#2a78d6", lw=1.5, ls=(0, (4, 3)))
ax.text(1.45, RTOL * 2.2, "rtol = 1e-9", color="#2a78d6", ha="right", fontsize=9)
ax.annotate(
    "$.output['price']\n$.output['value']\nin both records",
    xy=(0.85, np.min(act360)),
    xytext=(0.45, 3e-6),
    family="monospace",
    fontsize=9,
    color="#0b0b0b",
    arrowprops={"arrowstyle": "->", "color": "#52514e"},
)
ax.set_yscale("log")
ax.set_xticks([0, 1], ["numerical noise\n(passes)", "ACT/365 to ACT/360\n(differs)"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(1e-13, 1e-1)
ax.set_ylabel("|relative change in price|, per position")
ax.set_title("Same book, two changes: tolerance separates them", loc="left")
ax.legend(loc="center left")
fig.tight_layout()

# %%
# ``code_drift`` is false: ``value_book``'s own source is unchanged, so only the values
# could have caught this.

assert not any(case.code_drift for case in report.cases)

# %%
# Accepting a deliberate change
# -----------------------------
#
# Suppose ACT/360 was the right call after all. Accepting it is three steps: record the
# scenarios again, delete the old records, and prune the side files they leave.
#
# Records are never replaced, and a process records each call once, so re-recording
# happens in a fresh process, as ``REGRESSION_RECORDS=1 python run_pipeline.py``
# would (:ref:`sphx_glr_auto_examples_02_workflow_05_accepting_a_change.py` has the
# details). It inherits ``REGRESSION_RECORDS_STORE``, so the new records land beside
# the old ones.

result = subprocess.run(
    [
        sys.executable,
        "-c",
        "from regression_records._examples import revaluation\n"
        "revaluation.DAYS_PER_YEAR = 360.0  # the edited revaluation.py\n"
        "revaluation.record_scenarios()\n",
    ],
    env={**os.environ, "REGRESSION_RECORDS": "1"},
    capture_output=True,
    text=True,
    timeout=120,
    check=True,
)
with changed(DAYS_PER_YEAR=360.0):
    report = replay(tolerance=TOLERANCE)
print(report.counts())
assert report.counts()["passed"] == 2 and report.counts()["differs"] == 2

# %%
# The old records now fail, as they should: they and the code disagree. Deleting them is
# the deliberate act, done by hand, since records are never deleted for you; ``prune``
# then clears the side files no record refers to any more.
#
# In a real repository the store is committed, so ``git diff`` on it is the review: the
# old records out, the new ones in.

for case in report.failures:
    (STORE / case.record_path).unlink()
pruned = RecordStore(STORE).prune(dry_run=False)
print("pruned:", [p.name for p in pruned])
with changed(DAYS_PER_YEAR=360.0):
    report = replay(tolerance=TOLERANCE)
print(report.summary())
assert report.passed and len(report.cases) == 2

# %%
# A stale record left in the store keeps failing replay, which is accurate, until
# someone decides which of it and the code is right.
#
# In a test file
# --------------
#
# In a suite the replay is one line. ``replay_tests``
# (:ref:`sphx_glr_auto_examples_04_replay_in_practice_14_replay_tests.py`) makes each
# record its own pytest test, so a failure names the scenario that moved:
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

# %%
# **Next:** :ref:`sphx_glr_auto_examples_04_replay_in_practice_14_replay_tests.py`
# turns a store into one pytest test per record.
