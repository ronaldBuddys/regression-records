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

# %%
# Setup: the code under test
# --------------------------
#
# Replay finds a recorded function again by importing it by module and name, so the code
# under test has to live in a module. A function defined on this page belongs to
# ``__main__``, which cannot be imported again, and decorating one there raises
# ``TypeError``. In your project that module is your own package; here the page writes
# one, ``stats.py``, to a fresh temporary directory and keeps the store beside it, so
# the page - and the notebook it downloads as - needs nothing else.
#
# ``store`` is mandatory. There is no default location, because recording somewhere
# implicit would be worse than being asked.

import json
import sys
import tempfile
from pathlib import Path

WORKDIR = Path(tempfile.mkdtemp(prefix="regression-records-"))
(WORKDIR / "stats.py").write_text('''
from pathlib import Path

import numpy as np

from regression_records import record_func

STORE = Path(__file__).parent / "store"


@record_func(STORE)
def summarize(values: list[float], *, scale: float = 1.0) -> dict:
    """Return the size, mean and scaled values of ``values``."""
    scaled = np.asarray(values) * scale
    return {"n": int(scaled.size), "mean": float(scaled.mean()), "values": scaled}
''')
sys.path.insert(0, str(WORKDIR))

import stats

STORE = stats.STORE
print("store:", STORE)


def show_tree(root: Path) -> None:
    """Print every file in the store, relative to its root."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(" ", path.relative_to(root))


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
# From Python, ``enable()`` does the same. This page switches recording on for the whole
# script; :ref:`sphx_glr_auto_examples_02_workflow_04_modes_and_new_cases.py` covers the
# switch in full.

from regression_records import enable, replay

enable()

# %%
# Call it
# -------

result = stats.summarize([1.0, 2.0, 3.0, 4.0])
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
# ``stats.summarize`` is filed under ``stats/summarize/``.

record_path = next(STORE.rglob("rec-*.json"))
record = json.loads(record_path.read_text())
print(
    json.dumps(
        {k: record[k] for k in ("identity", "input_hash", "invocation", "output")},
        indent=2,
    )
)

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
#   reports the edit as code drift instead
#   (:ref:`sphx_glr_auto_examples_03_what_gets_recorded_08_exceptions_and_drift.py`).

assert record["invocation"]["kwargs"]["scale"] == 1.0
assert record_path.name.startswith(f"rec-{record['input_hash'][:16]}")

# %%
# One record per distinct call
# ----------------------------
#
# A function accumulates one record per distinct invocation. A call repeated with the
# same arguments is the same case, and writes nothing.

calls = [
    ([1.0, 2.0, 3.0, 4.0], {"scale": 2.0}),
    ([10.0, 20.0], {}),
    ([1.0, 2.0, 3.0, 4.0], {}),  # the very first call again
]
records_after = [len(list(STORE.rglob("rec-*.json")))]
for values, kwargs in calls:
    stats.summarize(values, **kwargs)
    records_after.append(len(list(STORE.rglob("rec-*.json"))))

assert records_after == [1, 2, 3, 3]
show_tree(STORE)

# %%
# Four calls, three records: the repeat added nothing.

import matplotlib.pyplot as plt

labels = ["[1, 2, 3, 4]", "[1, 2, 3, 4]\nscale=2", "[10, 20]", "[1, 2, 3, 4]\nagain"]
new = [b - a for a, b in zip([0, *records_after], records_after, strict=False)]
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.bar(
    labels,
    records_after,
    color=["#2a78d6" if n else "#c3c2b7" for n in new],
    width=0.6,
)
for x, (total, added) in enumerate(zip(records_after, new, strict=True)):
    ax.annotate(
        "new record" if added else "already recorded",
        (x, total),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        color="#52514e",
    )
ax.set_ylabel("records in the store")
ax.set_xlabel("call to summarize(...)")
ax.set_yticks(range(4))
ax.set_ylim(0, 3.6)
ax.set_title("Each distinct call is recorded once")
fig.tight_layout()

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
for case in report.cases:
    print(f"  {case.status.value:<8} {case.name}  record {case.record_id[:8]}")

# %%
# ``report.passed`` is the one-line verdict, the thing a test asserts on:

assert report.passed, report.summary()
assert report.counts()["passed"] == 3

# %%
# **Next:** :ref:`sphx_glr_auto_examples_01_basics_02_catching_a_regression.py` breaks
# the code and reads the failure.
