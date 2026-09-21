"""
replay : catching a regression
==============================

This example records a baseline, breaks the code behind the recorded function, and
reads the failure ``replay`` reports.

The break is deliberately *not* in the decorated function itself - it is in a helper it
calls, which is what a real regression usually looks like: something underneath changed
and the function on top now answers differently.
"""

import inspect
import sys
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import numpy as np
from _shared import reports
from _shared.demo import fresh_store, record_paths, show_cases, show_result

from regression_records import enable, replay

STORE = fresh_store(reports.STORE.name)

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# The code under test
# -------------------
#
# ``report`` is recorded; ``weight`` is the helper beneath it. Both live in
# ``examples/_shared/reports.py``, because replay has to import the function by name.

print(inspect.getsource(reports.weight))
print(inspect.getsource(reports.report))

# %%
# Record a baseline
# -----------------

baseline = reports.report([1.0, 2.0, 3.0])
reports.report([5.0, 5.0])
print(baseline)
print("records:", len(record_paths(STORE)))

# %%
# Break it
# --------
#
# A new ``weight`` - a plausible-looking change that shifts the numbers, drops a field,
# and renames another. Assigning it over the module's ``weight`` stands in for editing
# ``reports.py``; the ``finally`` puts the original back, so that nothing else run in
# this process sees the broken one.


def broken_weight(values):
    weighted = np.asarray(values) * 2.1
    return {"total": float(weighted.sum()), "series": weighted, "name": "weighted"}


original_weight = reports.weight
reports.weight = broken_weight
try:
    report = replay(STORE)
finally:
    reports.weight = original_weight

print("passed:", report.passed)
print(report.counts())
show_cases(report)

# %%
# Both records now differ. ``report.counts()`` tallies every status, so a script can
# tell "the answers changed" (``differs``) from "the call could not run"
# (``raised``, ``not_importable``, ...) without parsing text.
#
# Reading one case
# ----------------
#
# ``report.failures`` holds the cases that did not pass. Each carries its
# ``ComparisonResult``, which collects **every** difference rather than stopping at the
# first, so one failure tells you everything that moved.

case = report.failures[0]
show_result(case.comparison)

# %%
# Reading that:
#
# * ``$.output.total`` - a plain numeric difference, with the tolerance it was judged
#   against and the actual size of the gap.
# * ``$.output.series`` - the array differs, reported by *position* rather than by
#   dumping two arrays at you.
# * ``$.output.peak`` / ``$.output.name`` - a key that vanished and a key that appeared.
#   A renamed field shows up as this pair.
# * ``$.output.label`` is not mentioned at all, because it did not change. Only
#   differences are reported.
#
# The ``$`` path grammar is public: ``$.output`` for the return value, ``$.state_before`` /
# ``$.state_after`` for a method's object, ``.key`` for a mapping entry or attribute,
# ``[0]`` for a sequence element, ``['price']`` for a dataframe column. Those same strings
# key ``ToleranceConfig.path_overrides`` in example 3.

# %%
# The same, as text
# -----------------
#
# ``case.describe()`` is that case as one readable block: the status, the function, the
# record it replayed, and the diffs. ``report.summary()`` is the counts followed by one
# such block per failure - the message to put in an assertion.

print(case.describe())
print()
print(report.summary(max_diffs=2))

# %%
# ``code_drift`` is ``False`` on both cases: ``report``'s own source did not change, only
# the helper beneath it. Code drift tracks the recorded function's source alone, so a
# regression like this one is caught by the values, never by the hash.

print("code drift:", [c.code_drift for c in report.cases])

# %%
# Nothing was written
# -------------------
#
# Replay only reads. The baseline is still the baseline, and with the original
# ``weight`` back in place the store replays clean again.

print("records:", [p.name for p in record_paths(STORE)])
print(replay(STORE).summary())
