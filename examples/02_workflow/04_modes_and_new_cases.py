"""
``REGRESSION_RECORDS`` : switching recording on and off
=======================================================

A decorated function records only while recording is switched on. Otherwise its
wrapper is a pass-through that costs one check per call, so the decorator can stay on
production code.

The switch is the environment variable ``REGRESSION_RECORDS``, read afresh on every
call. ``enable()``, ``disable()`` and ``with recording():`` set it from Python. This
example tries each, then shows the two things that override it: ``enabled=False`` on a
decorator, and a running replay.

The switch is process-wide, and the other examples run in this same process when the
docs are built, so every section below puts it back the way it found it.
"""

import inspect
import os
import sys
from contextlib import contextmanager
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared import switching
from _shared.demo import fresh_store, show_cases, show_counts

from regression_records import RecordStore, disable, enable, recording, replay

STORE = fresh_store(switching.STORE.name)

print(inspect.getsource(switching))

# %%
# The switch is an environment variable
# -------------------------------------
#
# ``switched`` sets ``REGRESSION_RECORDS`` to a given value for a block, and puts back
# whatever was there before, the way ``recording()`` does.


@contextmanager
def switched(value):
    """Set ``REGRESSION_RECORDS`` to ``value`` (None: unset) for the block."""
    previous = os.environ.get("REGRESSION_RECORDS")
    if value is None:
        os.environ.pop("REGRESSION_RECORDS", None)
    else:
        os.environ["REGRESSION_RECORDS"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("REGRESSION_RECORDS", None)
        else:
            os.environ["REGRESSION_RECORDS"] = previous


# %%
# Unset, it is off. ``1``, ``true``, ``yes`` and ``on`` switch it on, in any case;
# empty, ``0``, ``false``, ``no`` and ``off`` switch it off. Each value below calls
# ``area`` with a different width, so the widths that were recorded say which values
# switched it on.

values = [None, "1", "TRUE", "yes", "on", "", "0", "false", "no", "off"]
for width, value in enumerate(values, start=1):
    with switched(value):
        switching.area(width, 1.0)

store = RecordStore(STORE)
ctx = store.deserialization_context(switching.area.identity)
recorded = sorted(
    int(record.load_args(ctx)[0])
    for record in store.list_records(switching.area.identity)
)
print("recorded widths:", recorded)
print("values that switched it on:", [values[w - 1] for w in recorded])

# %%
# In practice the variable is set for the one run that should record, and left unset
# everywhere else:
#
# .. code-block:: bash
#
#    REGRESSION_RECORDS=1 python run_pipeline.py
#    REGRESSION_RECORDS=1 pytest
#
#
# Being an environment variable, it reaches subprocesses and CI jobs without a code
# change.
#
# A value it does not recognise
# -----------------------------
#
# A typo must not leave recording silently off, so anything else raises
# ``ValueError`` on the next decorated call.

with switched("maybe"):
    try:
        switching.area(2.0, 3.0)
    except ValueError as e:
        print(f"ValueError: {e}")

# %%
# ``enable()`` and ``disable()``
# ------------------------------
#
# From Python, ``enable()`` sets ``REGRESSION_RECORDS=1`` and ``disable()`` removes it.
# Being process-wide, they suit a script that records from start to finish. The other
# examples call ``enable()`` at the top for exactly that reason.
#
# This one is shown inside ``switched``, only so that what it changes is put back.

with switched(None):
    disable()
    switching.area(20.0, 1.0)  # a plain call
    enable()
    switching.area(30.0, 1.0)  # recorded
    disable()
    switching.area(40.0, 1.0)  # a plain call again

# One more area record than before: the width-30 call.
show_counts(STORE)

# %%
# ``with recording():``
# ---------------------
#
# Switches recording on for one block, and restores the previous state on the way out,
# even if the block raises. ``recording(False)`` switches it off for a block instead.
#
# Recording ``floor_area`` records the ``area`` calls it makes as well: each decorated
# function records its own calls, wherever they come from.

with recording():
    switching.floor_area([(4.0, 5.0), (3.0, 3.0)])
    with recording(False):
        switching.floor_area([(10.0, 10.0)])  # not recorded

show_counts(STORE)

# %%
# ``enabled=False``: off whatever the switch says
# -----------------------------------------------
#
# ``perimeter`` is decorated with ``enabled=False``. It never records, so its calls are
# plain calls even inside ``recording()``. Use it to leave a decorator in place while a
# function's records are not wanted.

with recording():
    print("perimeter:", switching.perimeter(2.0, 3.0))

show_counts(STORE)

# %%
# Replay never records
# --------------------
#
# Replay calls each recorded function again, and those calls may reach decorated
# functions. Nothing records while a replay runs, whatever the switch says: a record
# written then would capture today's behavior as if it were a baseline.
#
# To show it, this replay runs with recording switched on, and with ``WALL`` changed so
# that ``floor_area`` calls ``area`` with widths nobody recorded. The ``floor_area``
# record differs, and no new ``area`` record appears.

with recording():
    switching.WALL = 0.5
    try:
        report = replay(STORE)
    finally:
        switching.WALL = 0.0

show_cases(report)
print()
show_counts(STORE)
