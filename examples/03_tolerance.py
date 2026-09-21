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
# # 3. Tolerance
#
# Floating point results drift: a BLAS update, a different summation order, a new CPU.
# `ToleranceConfig` decides how much drift is noise and how much is a regression.
#
# It applies to **values only**. Matching a call to its record is always exact - inputs
# that differ at all are a different case, never a nearly-matching one.

# %%
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)

import numpy as np
from _shared.demo import fresh_store, show_result

from regression_records import (
    RegressionMismatchError,
    ToleranceConfig,
    compare,
    record_func,
)

# %% [markdown]
# ## `compare()` on its own
#
# `compare()` is a pure function - it never raises, and it is the same code path the
# decorators use. That makes it the quickest way to see what a tolerance actually does.
#
# The default is `rtol=1e-7, atol=0.0`: two values are close when
# `abs(actual - expected) <= atol + rtol * abs(expected)`, the same rule as
# `numpy.isclose`.

# %%
print("1e-9 relative drift:", compare(1.0, 1.000000001).passed)
print("1e-3 relative drift:", compare(1.0, 1.001).passed)

# %% [markdown]
# Loosen it when the drift is expected:

# %%
loose = ToleranceConfig(rtol=1e-2)
print("1e-3 drift at rtol=1e-2:", compare(1.0, 1.001, loose).passed)

# %% [markdown]
# ## `atol` is for values near zero
#
# `rtol` is a fraction *of the expected value*, so it collapses to nothing as the expected
# value approaches zero: no relative tolerance makes `0.0` and `1e-12` close. `atol` is
# the absolute floor that does.

# %%
print(
    "0.0 vs 1e-12, rtol only:  ", compare(0.0, 1e-12, ToleranceConfig(rtol=1e-2)).passed
)
print(
    "0.0 vs 1e-12, with atol:  ", compare(0.0, 1e-12, ToleranceConfig(atol=1e-9)).passed
)

# %% [markdown]
# ## NaN, infinity, and dtype
#
# * `equal_nan=True` (the default) treats NaN in the same position on both sides as equal.
#   Recorded output is full of NaN in practice, and NaN != NaN would fail every time.
# * Infinities compare exactly - `inf` and `inf` match, `inf` and `1e308` do not, whatever
#   the tolerance.
# * `strict_dtype=True` (the default) makes an array's dtype part of the comparison. A
#   result that quietly became `float32`, or an integer column that became floats, is a
#   real change even when the numbers print the same.

# %%
nan_pair = (np.array([1.0, np.nan]), np.array([1.0, np.nan]))
print("NaN, equal_nan=True: ", compare(*nan_pair).passed)
print(
    "NaN, equal_nan=False:", compare(*nan_pair, ToleranceConfig(equal_nan=False)).passed
)

show_result(compare(np.array([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32)))
print()
print(
    "same, strict_dtype=False:",
    compare(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0], dtype=np.float32),
        ToleranceConfig(strict_dtype=False),
    ).passed,
)

# %% [markdown]
# ## Per-path overrides
#
# One tolerance for a whole result is usually too blunt: a Monte Carlo estimate needs
# room, an identifier alongside it needs none. `path_overrides` maps a path prefix to its
# own config, and the longest matching prefix wins.
#
# The keys use the same path grammar the diffs report - `$.output` for the return value,
# `.key` for a mapping entry, `[0]` for a sequence element, `['price']` for a dataframe
# column.

# %%
tolerance = ToleranceConfig(
    rtol=1e-9,  # strict by default
    path_overrides={
        "$.output.estimate": ToleranceConfig(rtol=0.05),  # 5% is fine here
    },
)

expected = {"estimate": 100.0, "count": 1000.0}
actual = {"estimate": 103.0, "count": 1000.0001}

show_result(compare(expected, actual, tolerance, root="$.output"))

# %% [markdown]
# The 3% move in `estimate` passed under its override; the far smaller move in `count`
# failed against the strict default. Note the reported `rtol` on the diff is the one
# actually applied, so a surprising pass or fail says which rule judged it.

# %% [markdown]
# ## On a decorator
#
# Exactly the same object, passed as `tolerance=`.

# %%
STORE = fresh_store("03_tolerance")
DRIFT = 0.0


@record_func(STORE, tolerance=ToleranceConfig(rtol=1e-3))
def simulate(seed):
    rng = np.random.default_rng(seed)
    return {"draws": rng.normal(size=4) * (1 + DRIFT), "seed": seed}


simulate(7)  # records the baseline

DRIFT = 1e-5  # a relative perturbation well inside rtol=1e-3
simulate(7)
print("passed within tolerance:", simulate.last_result.passed)

DRIFT = 0.5  # a real change

try:
    simulate(7)
except RegressionMismatchError as e:
    print(e.result.summary())

# %% [markdown]
# The perturbation inside the tolerance passed silently; the one outside it raised. That
# is the whole job of `tolerance`: draw the line between the drift you have decided not to
# care about and the change you want to hear about.
#
# One caution. A tolerance loose enough to hide the noise is also loose enough to hide a
# small real regression, and it is easier to widen a tolerance than to explain later why a
# suite stopped catching things. Prefer a tight default with narrow `path_overrides` where
# they are earned.
