"""
``RecordStore``, ``replay`` : re-recording after an intended change
===================================================================

Sooner or later a replay failure is *correct*: you changed the behavior on purpose, and
the record is what is out of date. Accepting the change takes three steps: re-record,
delete the old record by hand, and prune the data files it leaves behind.

Records are never replaced. Recording changed behavior adds a new record beside the old
one, and recording unchanged behavior writes nothing at all. So the store only changes
when behavior does, and removing a record is always a deliberate act.
"""

# %%
# Setup: where records go
# -----------------------
#
# The code under test reads its store from the ``REGRESSION_RECORDS_STORE`` environment
# variable when it is imported, so the variable is set first
# (:ref:`sphx_glr_auto_examples_01_basics_01_first_record.py` says why).

import os
import tempfile

# In a project: REGRESSION_RECORDS_STORE=tests/regression_data in a .env file, loaded
# with python-dotenv's load_dotenv(). Here: a fresh temporary directory, empty each run.
os.environ["REGRESSION_RECORDS_STORE"] = tempfile.mkdtemp(prefix="regression-records-")

# %%
# Setup: the code under test
# --------------------------
#
# Replay imports a recorded function by module and name, so the code under test lives in
# a module, ``fees``, imported from ``regression_records._examples`` - support code that
# ships with the package so that a page can import real code rather than define it in a
# cell (:ref:`sphx_glr_auto_examples_01_basics_01_first_record.py` says more). In your
# project, this is your own package. Its source, printed below:
#
# ``commission`` returns an array, so its record has a side file in ``data/``. ``RATE``
# is the commission rate; raising it below stands in for an intended edit to the file.

import inspect
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from regression_records._examples import fees

print(inspect.getsource(fees))

# %%

STORE = fees.STORE
print("store:", STORE)


def show_tree(root: Path) -> None:
    """Print every file in the store, relative to its root."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(" ", path.relative_to(root))


def run_in_fresh_process(code: str, **env: str) -> None:
    """Run ``code`` in a new Python process, and print its output.

    The new process shares nothing with this one: not its module settings, and not its
    memory of which calls it has already recorded. It inherits this one's environment,
    so ``fees`` there records into the same store, as a re-run of a script would.
    """
    result = subprocess.run(  # noqa: S603 - this interpreter, running this page's code
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    print(result.stdout, end="")
    assert result.returncode == 0, result.stderr


from regression_records import RecordStore, enable, replay

# %%
# Recording is off unless switched on; this page switches it on for the whole script.

enable()

# %%
# Record a baseline
# -----------------

AMOUNTS = [100.0, 250.0, 1000.0]
print("baseline:", fees.commission(AMOUNTS))
show_tree(STORE)

# Replay status counts after each step, for the figure at the end.
steps: dict[str, dict[str, int]] = {"recorded": replay(STORE).counts()}

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
steps["code changed"] = report.counts()
assert report.counts()["differs"] == 1
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

files_before = sorted(STORE.rglob("*"))
with edited():
    fees.commission(AMOUNTS)
assert sorted(STORE.rglob("*")) == files_before
show_tree(STORE)

# %%
# So re-record the way you would for real, in a fresh process with
# ``REGRESSION_RECORDS=1``, as in ``REGRESSION_RECORDS=1 python run_pipeline.py``.
# ``run_in_fresh_process`` runs a few lines of Python in one; being an installed module,
# ``fees`` is importable there without any extra setup.

RERECORD = """
from regression_records._examples import fees
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
steps["re-recorded"] = report.counts()
for case in report.cases:
    print(f"  {case.status.value:<8} {case.name}  record {case.record_id[:8]}")
assert report.counts()["passed"] == 1 and report.counts()["differs"] == 1

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
deleted = store.prune(dry_run=False)
print("deleted:     ", [path.name for path in deleted])
assert len(deleted) == 1
show_tree(STORE)

with edited():
    report = replay(STORE)
steps["old record deleted"] = report.counts()
print(report.summary())
assert report.passed

# %%
# The three steps at a glance: the change makes the one record fail, re-recording adds
# a passing record beside it, and deleting the old one leaves a store that agrees with
# the code.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.0, 3.4))
names = list(steps)
passed = [steps[name]["passed"] for name in names]
differs = [steps[name]["differs"] for name in names]
ax.bar(names, passed, color="#0ca30c", width=0.55, label="passed")
ax.bar(names, differs, bottom=passed, color="#d03b3b", width=0.55, label="differs")
for x, (p, d) in enumerate(zip(passed, differs, strict=True)):
    ax.text(x, p + d + 0.05, f"{p} passed, {d} differs", ha="center", color="#52514e")
ax.set_ylabel("records replayed")
ax.set_yticks([0, 1, 2])
ax.set_ylim(0, 2.5)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
ax.set_title("Accepting an intended change", loc="left")
fig.tight_layout()

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
assert before == after, "the store changed"

# %%
# **Next:** :ref:`sphx_glr_auto_examples_02_workflow_06_run_manifests.py` - a record of
# each recording run itself.
