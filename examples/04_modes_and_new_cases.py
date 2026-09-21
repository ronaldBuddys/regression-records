"""
Run modes and unseen inputs
===========================

Two settings decide what a call does when it finds no record for its arguments:

* ``mode`` (``RunMode``) - may this decoration create new recorded cases at all?
* ``on_new_case`` (``OnNewCase``) - given that it may, what should happen to inputs
  nothing was recorded for?

A third setting, ``overwrite``, answers a different question entirely - whether an
*existing* record may be replaced - and is example 5.
"""

import sys
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared.demo import fresh_store, record_paths

from regression_records import (
    OnNewCase,
    RecordNotFoundError,
    RunMode,
    UnmatchedInvocationError,
    record_func,
)

STORE = fresh_store("04_modes_and_new_cases")

# %%
# ``RunMode.BOTH`` - the default
# ------------------------------
#
# Validate a case that was recorded before; record one that was not. This is what you
# want while developing: run your code, and the store fills in behind you.


@record_func(STORE)
def area(width, height):
    return width * height


area(2, 3)  # new case -> recorded
area(2, 3)  # known case -> validated
area(4, 5)  # new case -> recorded
print("records:", len(record_paths(STORE)))

# %%
# ``RunMode.VALIDATE_ONLY`` - for CI
# ----------------------------------
#
# Never invents a baseline. This is the mode that makes a regression suite mean something:
# a test run cannot quietly record whatever the code does today and then pass.
#
# (``name="area"`` re-points a differently-named function at the identity recorded above, so
# one file can show several settings against one history. It also means the source behind
# that identity keeps changing, which is why a ``CodeDriftWarning`` appears - see example 7.)


@record_func(STORE, name="area", mode=RunMode.VALIDATE_ONLY)
def checked_area(width, height):
    return width * height


print("known case:", checked_area(2, 3))

try:
    checked_area(9, 9)  # never recorded
except UnmatchedInvocationError as e:
    print(f"\nUnmatchedInvocationError: {e}")

# %%
# The error distinguishes two situations that need different fixes:
#
# * ``UnmatchedInvocationError`` - this function has history, but not for *these*
#   arguments. Usually a test that gained a new case; record it deliberately.
# * ``RecordNotFoundError`` - this function has no history at all. Usually a store that
#   was not committed, or a function that moved module and left its records behind.


@record_func(STORE, mode=RunMode.VALIDATE_ONLY)
def never_recorded(x):
    return x


try:
    never_recorded(1)
except RecordNotFoundError as e:
    print(f"RecordNotFoundError: {e}")

# %%
# ``on_new_case``
# ---------------
#
# Left unset it follows the mode: record the new case, or - under ``VALIDATE_ONLY``, where
# an unrecorded input is exactly what you want surfaced - raise. Set it explicitly to
# override that.
#
# ``OnNewCase.IGNORE`` calls through and does nothing: no record, no comparison, no error.
# It is the pragmatic setting for a validating run over inputs that vary from run to run,
# where only the cases you have recorded are worth checking.


@record_func(
    STORE, name="area", mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.IGNORE
)
def lenient_area(width, height):
    return width * height


print("unrecorded input passes through:", lenient_area(9, 9))
print("last_result (no comparison ran):", lenient_area.last_result)
print("records still:", len(record_paths(STORE)))

# %%
# ``OnNewCase.ERROR`` is the same refusal under a mode that would otherwise record - "this
# function's cases are fixed, tell me if a new one appears".


@record_func(STORE, name="area", on_new_case=OnNewCase.ERROR)
def fixed_cases(width, height):
    return width * height


try:
    fixed_cases(11, 11)
except UnmatchedInvocationError as e:
    print(f"UnmatchedInvocationError: {e}")

# %%
# ``RunMode.RECORD_ONLY``
# -----------------------
#
# Always record, never compare - for seeding a store, or re-seeding one after a
# deliberate rewrite. Note that with the default ``overwrite=OverwritePolicy.NEVER`` an
# *existing* case is left alone rather than rewritten: history stays immutable unless you
# say otherwise.


@record_func(STORE, mode=RunMode.RECORD_ONLY)
def seeded(x):
    return x * 10


seeded(1)
seeded(2)
seeded(1)  # already recorded; not compared, and not rewritten
print(
    "records for `seeded`:",
    len(list((STORE / "__main__" / "seeded").rglob("rec-*.json"))),
)

# %%
# The one contradiction
# ---------------------
#
# ``VALIDATE_ONLY`` with an explicit ``on_new_case=OnNewCase.RECORD`` asks for two opposite
# things. It raises when the decorator is applied - not weeks later, on the first call
# that happens to reach an input nobody recorded.

try:

    @record_func(STORE, mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.RECORD)
    def contradictory(x):
        return x

except ValueError as e:
    print(f"ValueError at decoration time: {e}")

# %%
# Choosing a mode from the environment
# ------------------------------------
#
# The decorator's arguments are ordinary Python, so the usual way to run one suite two
# ways is to compute them:
#
# .. code-block:: python
#
#    import os
#
#    MODE = RunMode.VALIDATE_ONLY if os.environ.get("CI") else RunMode.BOTH
#
#    @record_func("tests/regression_data", mode=MODE)
#    def score(frame):
#        ...
#
#
# Locally the store fills in as you work; in CI nothing new is ever recorded and an
# unrecorded input is a failure.
