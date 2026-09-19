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
# # 5. Accepting an intentional change
#
# Sooner or later a mismatch is *correct*: you changed the behaviour on purpose and the
# recorded baseline is now the stale one. `OverwritePolicy` is how you say so.
#
# It is deliberately separate from `mode`. **`mode` decides whether a new case may be
# created; `overwrite` decides whether an existing record may be replaced.** They compose
# freely, and the useful combination is the one that looks contradictory at first:
# `VALIDATE_ONLY` with `IF_MISMATCH` re-baselines the cases you already have without
# inventing any new ones.

# %%
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd()))

from _shared.demo import fresh_store, record_paths, show_result
from regression_records import (
    OverwritePolicy,
    RegressionMismatchError,
    RunMode,
    record_func,
)

STORE = fresh_store("05_accepting_a_change")

RATE = 0.10


@record_func(STORE)
def commission(amount):
    return round(amount * RATE, 2)


print("baseline:", commission(250.0))

# %% [markdown]
# ## The rate changes on purpose

# %%
RATE = 0.12

try:
    commission(250.0)
except RegressionMismatchError as e:
    show_result(e.result)

# %% [markdown]
# Nothing was written. Under the default `OverwritePolicy.NEVER` a mismatch never touches
# the store, so a failing run cannot quietly repair itself into a passing one.

# %%
print("recorded output is still:", json.loads(record_paths(STORE)[0].read_text())["output"])

# %% [markdown]
# ## Accepting it
#
# Re-run the same code with `IF_MISMATCH`. The record is replaced with the new result, and
# **the call does not raise** - rewriting the baseline *is* the act of accepting, so
# reporting a failure as well would be reporting something you just approved.

# %%
@record_func(STORE, name="commission", overwrite=OverwritePolicy.IF_MISMATCH)
def commission_accepting(amount):
    return round(amount * RATE, 2)


print("returned:", commission_accepting(250.0))
print("\nwhat was accepted:")
show_result(commission_accepting.last_result)

# %% [markdown]
# The diffs are still on `last_result`, so an accepting run can print exactly what it
# approved. And because records are JSON on disk, the same change shows up in
# `git diff` - which is where a reviewer should be looking at it.
#
# The new baseline sticks:

# %%
print("validates against the new baseline:", commission(250.0))
print("passed:", commission.last_result.passed)
print("record count unchanged:", len(record_paths(STORE)))

# %% [markdown]
# ## `ALWAYS`
#
# Rewrites the matched record on every call, regardless of whether anything differed. Use
# it to re-baseline a whole suite after a change you have already decided is correct - a
# unit change, a new library version - where checking each case individually is not the
# point.
#
# `ALWAYS` never reports anything, so a run under it tells you nothing about correctness.

# %%
RATE = 0.15


@record_func(STORE, name="commission", mode=RunMode.VALIDATE_ONLY, overwrite=OverwritePolicy.ALWAYS)
def commission_rebaseline(amount):
    return round(amount * RATE, 2)


commission_rebaseline(250.0)
print("re-baselined to:", commission(250.0))

# %% [markdown]
# Pairing `ALWAYS` with `VALIDATE_ONLY` is the safer form of a re-baselining run: every
# case you already have is refreshed, and an argument set nobody recorded still raises
# instead of quietly joining the suite.
#
# ## Then take it back off
#
# `IF_MISMATCH` or `ALWAYS` left in place permanently is a suite that agrees with whatever
# the code currently does - which is not a regression suite. Accepting is a step you take
# once, deliberately, and then revert.
#
# ## One thing overwriting leaves behind
#
# Replacing a record does not delete the side files the old version referenced. They are
# content-addressed, so nothing breaks and nothing is corrupted - but arrays and
# dataframes from superseded baselines accumulate in `data/`, and there is no pruning
# helper yet. For most stores this is a few kilobytes; for one full of large frames,
# re-recording from scratch is currently the way to reclaim it.

# %%
import numpy as np

SHIFT = 0.0


@record_func(STORE, overwrite=OverwritePolicy.IF_MISMATCH)
def curve(n):
    return np.arange(n) + SHIFT


curve(4)  # records, writing one .npy
for SHIFT in (1.0, 2.0, 3.0):  # each accepted change writes another
    curve(4)

print("records for `curve`:", len(list((STORE / "__main__" / "curve" / "records").glob("*.json"))))
print("data files for `curve`:", len(list((STORE / "__main__" / "curve" / "data").glob("*"))))
