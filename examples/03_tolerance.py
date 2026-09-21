"""
ToleranceConfig : replaying within tolerances
=============================================

Floating point results drift: a BLAS update, a different summation order, a new CPU.
``ToleranceConfig`` decides how much drift is noise and how much is a regression, and
``replay`` applies it when it compares a replayed call with its record.

It applies to **values only**. Which function and which arguments a record replays is
never approximate.
"""

import sys
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import numpy as np
from _shared import sampling
from _shared.demo import fresh_store, show_cases, show_result

from regression_records import ToleranceConfig, compare, enable, replay

# %%
# ``compare()`` on its own
# ------------------------
#
# ``compare()`` is a pure function - it never raises, and it is the same code path
# ``replay`` uses. That makes it the quickest way to see what a tolerance actually does.
#
# The default is ``rtol=1e-7, atol=0.0``: two values are close when
# ``abs(actual - expected) <= atol + rtol * abs(expected)``, the same rule as
# ``numpy.isclose``.

print("1e-9 relative drift:", compare(1.0, 1.000000001).passed)
print("1e-3 relative drift:", compare(1.0, 1.001).passed)

# %%
# Loosen it when the drift is expected:

loose = ToleranceConfig(rtol=1e-2)
print("1e-3 drift at rtol=1e-2:", compare(1.0, 1.001, loose).passed)

# %%
# ``atol`` is for values near zero
# --------------------------------
#
# ``rtol`` is a fraction *of the expected value*, so it collapses to nothing as the expected
# value approaches zero: no relative tolerance makes ``0.0`` and ``1e-12`` close. ``atol`` is
# the absolute floor that does.

print(
    "0.0 vs 1e-12, rtol only:  ", compare(0.0, 1e-12, ToleranceConfig(rtol=1e-2)).passed
)
print(
    "0.0 vs 1e-12, with atol:  ", compare(0.0, 1e-12, ToleranceConfig(atol=1e-9)).passed
)

# %%
# NaN, infinity, and dtype
# ------------------------
#
# * ``equal_nan=True`` (the default) treats NaN in the same position on both sides as equal.
#   Recorded output is full of NaN in practice, and NaN != NaN would fail every time.
# * Infinities compare exactly - ``inf`` and ``inf`` match, ``inf`` and ``1e308`` do not, whatever
#   the tolerance.
# * ``strict_dtype=True`` (the default) makes an array's dtype part of the comparison. A
#   result that quietly became ``float32``, or an integer column that became floats, is a
#   real change even when the numbers print the same.

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

# %%
# Per-path overrides
# ------------------
#
# One tolerance for a whole result is usually too blunt: a Monte Carlo estimate needs
# room, an identifier alongside it needs none. ``path_overrides`` maps a path prefix to its
# own config, and the longest matching prefix wins. An override changes only the fields
# it is given and inherits the rest from the config above it, so the ``estimate`` override
# below keeps the parent's ``atol``, ``equal_nan`` and ``strict_dtype``.
#
# The keys use the same path grammar the diffs report - ``$.output`` for the return value,
# ``.key`` for a mapping entry, ``[0]`` for a sequence element, ``['price']`` for a dataframe
# column.

tolerance = ToleranceConfig(
    rtol=1e-9,  # strict by default
    path_overrides={
        "$.output.estimate": ToleranceConfig(rtol=0.05),  # 5% is fine here
    },
)

expected = {"estimate": 100.0, "count": 1000.0}
actual = {"estimate": 103.0, "count": 1000.0001}

show_result(compare(expected, actual, tolerance, root="$.output"))

# %%
# The 3% move in ``estimate`` passed under its override; the far smaller move in ``count``
# failed against the strict default. Note the reported ``rtol`` on the diff is the one
# actually applied, so a surprising pass or fail says which rule judged it.

# %%
# Recording a baseline
# --------------------
#
# ``examples/_shared/sampling.py`` holds two recorded functions: ``simulate``, a Monte
# Carlo estimate, and ``tally``, a currency total that should be exact. Its ``DRIFT``
# setting perturbs every float they return, standing in for the noise a new machine or
# library version brings.
#
# Recording is off unless switched on; this example switches it on for the rest of the
# script.

enable()

STORE = fresh_store(sampling.STORE.name)

sampling.simulate(7)
sampling.simulate(11)
sampling.tally([19.99, 5.01, 100.0])


def replay_with_drift(drift, **replay_kwargs):
    """Replay the store while every float result is perturbed by ``drift``."""
    sampling.DRIFT = drift
    try:
        return replay(STORE, **replay_kwargs)
    finally:
        sampling.DRIFT = 0.0


# %%
# One tolerance for the whole replay
# ----------------------------------
#
# A relative drift of ``1e-5`` fails the default ``rtol=1e-7`` everywhere. Passed as
# ``tolerance=``, a looser config forgives it.

show_cases(replay_with_drift(1e-5))
print()
show_cases(replay_with_drift(1e-5, tolerance=ToleranceConfig(rtol=1e-3)))

# %%
# The tolerance belongs to the replay, not to the recording. The records hold values;
# how closely a later run must reproduce them is decided when it runs, so the same store
# can be checked strictly on the machine that recorded it and loosely elsewhere.
#
# Per function: ``tolerance_overrides``
# -------------------------------------
#
# ``rtol=1e-3`` is right for a Monte Carlo estimate and wrong for a currency total.
# ``tolerance_overrides`` maps a glob on ``"module:qualname"`` to its own config; the
# first pattern that matches wins, and every other function gets ``tolerance``.

report = replay_with_drift(
    1e-5,
    tolerance=ToleranceConfig(rtol=1e-9),
    tolerance_overrides={"_shared.sampling:sim*": ToleranceConfig(rtol=1e-3)},
)
show_cases(report)
print()
print(report.failures[0].describe())

# %%
# ``simulate`` passed under its override; ``tally`` was judged by the strict default and
# its drifted total is reported. ``count`` is an integer and is not mentioned: integers
# always compare exactly, whatever the tolerance.
#
# Per value: ``path_overrides`` on replay
# ---------------------------------------
#
# The two layers combine. ``tolerance_overrides`` picks a config per function, and that
# config's ``path_overrides`` pick values within the function's result. Here only the
# estimate is loose; the draws inherit the strict ``rtol=1e-9`` and fail.

report = replay_with_drift(
    1e-5,
    tolerance=ToleranceConfig(rtol=1e-9),
    tolerance_overrides={
        "_shared.sampling:simulate": ToleranceConfig(
            rtol=1e-9, path_overrides={"$.output.estimate": ToleranceConfig(rtol=1e-3)}
        )
    },
)
show_result(report.failures[0].comparison)

# %%
# A real change
# -------------
#
# Drift of 50% is not noise. Even the loosest config above reports it.

report = replay_with_drift(0.5, tolerance=ToleranceConfig(rtol=1e-3))
print(report.summary(max_diffs=2))

# %%
# The perturbation inside the tolerance passed silently; the one outside it failed. That
# is the whole job of ``tolerance``: draw the line between the drift you have decided not
# to care about and the change you want to hear about.
#
# One caution. A tolerance loose enough to hide the noise is also loose enough to hide a
# small real regression, and it is easier to widen a tolerance than to explain later why
# a suite stopped catching things. Prefer a tight default with narrow overrides - per
# function, then per path - where they are earned.
