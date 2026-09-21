"""
Re-recording : after an intended change
=======================================

Sooner or later a replay failure is *correct*: you changed the behavior on purpose, and
the record is what is out of date. Accepting the change takes three steps: re-record,
delete the old record by hand, and prune the data files it leaves behind.

Records are never replaced. Recording changed behavior adds a new record beside the old
one, and recording unchanged behavior writes nothing at all. So the store only changes
when behavior does, and removing a record is always a deliberate act.
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

from _shared import fees
from _shared.demo import fresh_store, run_in_fresh_process, show_cases, show_tree

from regression_records import RecordStore, enable, replay

STORE = fresh_store(fees.STORE.name)

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# Record a baseline
# -----------------
#
# ``commission`` lives in ``examples/_shared/fees.py``, where replay can import it. It
# returns an array, so its record has a side file in ``data/``.

print(inspect.getsource(fees.commission))

AMOUNTS = [100.0, 250.0, 1000.0]
print("baseline:", fees.commission(AMOUNTS))
show_tree(STORE)

# %%
# Change it on purpose
# --------------------
#
# The commission rate goes up from 10% to 12%. ``edited`` sets ``fees.RATE`` for a
# block, standing in for the edit to ``fees.py``, and puts it back afterwards.


@contextmanager
def edited():
    """Run the block as though ``fees.py`` now said ``RATE = 0.12``."""
    fees.RATE = 0.12
    try:
        yield
    finally:
        fees.RATE = 0.10


with edited():
    report = replay(STORE)
show_cases(report)
print()
print(report.failures[0].describe())

# %%
# That is the failure you expect. Replay compares with what was recorded, not with what
# is right, so it is for you to decide the new numbers are the ones you want.
#
# Re-record
# ---------
#
# Re-recording means running the same calls again with recording switched on. Doing it
# in this process writes nothing, though: **within one process a call is recorded
# once**, keyed on its arguments, so a call repeated in a loop is not written again and
# again. The store is unchanged:

with edited():
    fees.commission(AMOUNTS)
show_tree(STORE)

# %%
# So re-record the way you would for real, in a fresh process with
# ``REGRESSION_RECORDS=1``, as in ``REGRESSION_RECORDS=1 python run_pipeline.py``.
# ``run_in_fresh_process`` runs a few lines of Python in one, with ``_shared``
# importable.

RERECORD = """
from _shared import fees
fees.RATE = 0.12  # the edited fees.py
print("re-recorded:", fees.commission([100.0, 250.0, 1000.0]))
"""

run_in_fresh_process(RERECORD, REGRESSION_RECORDS="1")
show_tree(STORE)

# %%
# A second record, beside the first, with its own array in ``data/``. A record's id is
# derived from what the call did, so the changed behavior got a new file rather than
# replacing the old one.
#
# Both records now replay: the new one passes, and the old one still fails, because it
# and the code still disagree.

with edited():
    report = replay(STORE)
show_cases(report)

# %%
# Both records are of one call, so they share an input hash, the leading part of their
# filenames. Grouping cases by ``case.input_hash`` is how to spot a call with more than
# one record.

for case in report.cases:
    print(f"  {case.status.value:<8} input_hash {case.input_hash[:12]}")

# %%
# Delete the old record
# ---------------------
#
# Nothing deletes a record for you. The failing case names its file, relative to the
# store; delete it by hand, as you would with ``git rm``.

stale = STORE / report.failures[0].record_path
print("deleting", stale.relative_to(STORE))
stale.unlink()

# %%
# Its array is still in ``data/``, referenced by nothing. ``prune`` lists such files, and
# deletes them with ``dry_run=False``.

store = RecordStore(STORE)
print("unreferenced:", [path.name for path in store.prune()])
print("deleted:     ", [path.name for path in store.prune(dry_run=False)])
show_tree(STORE)

with edited():
    print(replay(STORE).summary())

# %%
# In a repository the store is committed, so the review of the change is the store's
# ``git diff``: one record out, one in.
#
# Re-recording unchanged behavior writes nothing
# ----------------------------------------------
#
# Run the recording again, in another fresh process. The call behaves exactly as its
# record says, so it lands on the same record id, the file is already there, and
# nothing is written. That also holds for parallel pytest-xdist workers recording the
# same call at once.

before = sorted(path.name for path in STORE.rglob("*") if path.is_file())
run_in_fresh_process(RERECORD, REGRESSION_RECORDS="1")
after = sorted(path.name for path in STORE.rglob("*") if path.is_file())
print("store unchanged:", before == after)
