"""
``record_func`` : recording a first call
========================================

``record_func`` needs two things: a function, and somewhere to keep its history.

While recording is switched on, each distinct set of arguments the decorated function
is called with is written to disk as a **regression record**: what went in, and what
came out. Nothing else about the function changes - arguments arrive untouched, the
return value passes through.

Recording is half of the job. The other half is ``replay``: later, after the code or
its environment has changed, it calls the function again on every recorded input and
reports whether it still gives the recorded answer.
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

from _shared import stats
from _shared.demo import fresh_store, show_cases, show_record, show_tree

from regression_records import enable, replay

STORE = fresh_store(stats.STORE.name)

# %%
# The decorated function
# ----------------------
#
# ``store`` is mandatory. There is no default location, because recording somewhere
# implicit would be worse than being asked.
#
# The function lives in ``examples/_shared/stats.py`` rather than in this script. Replay
# finds a recorded function again by importing it by name, and a function defined in a
# script belongs to ``__main__``, which cannot be imported again - so decorating one
# there raises ``TypeError``.

print(inspect.getsource(stats))

# %%
# Switch recording on
# -------------------
#
# Recording is off unless switched on, so a decorated function costs one check per call
# and the decorator can stay on production code. The usual switch is the environment
# variable, for the one run that should record:
#
# .. code-block:: bash
#
#    REGRESSION_RECORDS=1 python run_pipeline.py
#
#
# From Python, ``enable()`` does the same. This example switches recording on for the
# whole script; example 4 covers the switch in full.

enable()

# %%
# Call it
# -------

from _shared.stats import summarize

result = summarize([1.0, 2.0, 3.0, 4.0])
result

# %%
# What landed on disk
# -------------------
#
# One JSON record, plus a side file for the array. Arrays and dataframes are too big and
# too binary to sit inside JSON, so they are written next to it and referenced by a
# relative path - which is what lets the whole directory be moved or copied as a unit.

show_tree(STORE)

# %%
# The directory path is the function's identity: its module, then its qualified name.
# ``_shared.stats.summarize`` is filed under ``_shared/stats/summarize/``.

show_record(STORE, keys=["identity", "input_hash", "invocation", "output"])

# %%
# Worth noticing in that JSON:
#
# * ``invocation.args`` and ``invocation.kwargs`` hold the arguments *after* binding them
#   to the signature. ``scale`` was never passed, but it is recorded as ``1.0``, so
#   replay calls the function exactly as it was called, default included.
# * ``input_hash`` is a sha256 over that normalized argument tree. It is the leading part
#   of the record's filename, so the records of one call sit together.
# * ``identity.code_hash`` hashes the function's source. It is recorded but is not part
#   of the filename: editing the function does not orphan its history, and replay
#   reports the edit as code drift instead (example 7).

# %%
# Different arguments are a different case
# ----------------------------------------
#
# A function accumulates one record per distinct invocation.

summarize([1.0, 2.0, 3.0, 4.0], scale=2.0)
summarize([10.0, 20.0])
show_tree(STORE)

# %%
# Replaying the records
# ---------------------
#
# ``replay`` walks the store. For each record it imports the function by its recorded
# name, calls the function beneath the decorator on the recorded arguments, and compares
# what comes back with what was recorded. Nothing is written: replay only reads.
#
# Nothing has changed since the recording, so every case passes.

report = replay(STORE)
print(report.summary())
show_cases(report)

# %%
# ``report.passed`` is the one-line verdict, the thing a test asserts on:
#
# .. code-block:: python
#
#    assert report.passed, report.summary()
#
# Example 2 breaks the code and reads the failure; example 14 turns each record into its
# own pytest test.

assert report.passed, report.summary()
