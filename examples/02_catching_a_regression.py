# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Catching a regression
#
# Recording is only half of it. This example records a baseline, breaks the code behind
# the recorded function, and reads the failure.
#
# The break is deliberately *not* in the decorated function itself - it is in a helper it
# calls, which is what a real regression usually looks like: something underneath changed
# and the function on top now answers differently.

# %%
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)

import numpy as np
from _shared.demo import fresh_store, show_result

from regression_records import OnMismatch, RegressionMismatchError, record_func

STORE = fresh_store("02_catching_a_regression")


def weight(values):
    """The helper we are about to break."""
    return np.asarray(values) * 2.0


@record_func(STORE)
def report(values):
    weighted = weight(values)
    return {
        "total": float(weighted.sum()),
        "peak": float(weighted.max()),
        "series": weighted,
        "label": "weighted",
    }


baseline = report([1.0, 2.0, 3.0])
baseline

# %% [markdown]
# ## Break it
#
# A new `weight` - a plausible-looking change that shifts the numbers, drops a field, and
# renames another.


# %%
def weight(values):  # noqa: F811 - deliberately shadowing, to simulate a code change
    return np.asarray(values) * 2.1


@record_func(STORE, name="report")
def broken_report(values):
    weighted = weight(values)
    return {
        "total": float(weighted.sum()),
        "series": weighted,
        "label": "weighted",
        "name": "weighted",
    }


# %% [markdown]
# `name="report"` points this function at the identity the baseline was recorded under.
# In real use you would simply edit `report` in place; this file needs both versions alive
# at once to show them side by side.
#
# That is also why the calls below emit a `CodeDriftWarning`: two different function
# bodies are sharing one identity, which is exactly what that warning is for. Example 7
# covers it.
#
# ## The mismatch
#
# The comparison collects **every** difference rather than stopping at the first, so one
# failure tells you everything that moved.

# %%
try:
    broken_report([1.0, 2.0, 3.0])
except RegressionMismatchError as e:
    show_result(e.result)

# %% [markdown]
# Reading that:
#
# * `$.output.total` - a plain numeric difference, with the tolerance it was judged
#   against and the actual size of the gap.
# * `$.output.series` - the array differs, reported by *position* rather than by dumping
#   two arrays at you.
# * `$.output.peak` / `$.output.name` - a key that vanished and a key that appeared. A
#   renamed field shows up as this pair.
# * `$.output.label` is not mentioned at all, because it did not change. Only differences
#   are reported.
#
# The `$` path grammar is public: `$.output` for the return value, `$.state_before` /
# `$.state_after` for a method's object, `.key` for a mapping entry or attribute, `[0]`
# for a sequence element, `['price']` for a dataframe column. Those same strings key
# `ToleranceConfig.path_overrides` in example 3.

# %% [markdown]
# ## Not raising
#
# `on_mismatch=OnMismatch.RETURN` lets the call return normally and leaves the verdict on
# the wrapper. Useful when you want to collect mismatches across a whole run rather than
# stopping at the first, or when a mismatch is information rather than a failure.


# %%
@record_func(STORE, name="report", on_mismatch=OnMismatch.RETURN)
def quiet_report(values):
    weighted = weight(values)
    return {
        "total": float(weighted.sum()),
        "series": weighted,
        "label": "weighted",
        "name": "weighted",
    }


returned = quiet_report([1.0, 2.0, 3.0])
print("the call returned normally:", type(returned).__name__)
print("passed:", quiet_report.last_result.passed)
print("diff paths:", [d.path for d in quiet_report.last_result.diffs])

# %% [markdown]
# `last_result` is set the same way after a raising mismatch, so it is always available
# afterwards - `RegressionMismatchError.result` and `wrapper.last_result` are the same
# object.
#
# ## Nothing was written
#
# A mismatch under the default `OverwritePolicy.NEVER` does not touch the store: the
# baseline is still the baseline, and the side files for the new values were discarded
# rather than left behind unreferenced. Accepting a change is a deliberate act - example 5.

# %%
print("records:", [p.name for p in STORE.rglob("rec-*.json")])
print("data files:", sorted(p.name for p in STORE.rglob("*.npy")))
