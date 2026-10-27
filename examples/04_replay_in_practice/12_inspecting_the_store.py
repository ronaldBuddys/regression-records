"""
``RecordStore`` : inspecting records on disk
============================================

A store is JSON in directories - ``grep``, ``git diff`` and ``jq`` all work on it. But
there is also an API, for the times you want to walk every record the way ``replay``
does, read a recorded value back as a Python object, find out what a function's history
actually covers, or move records after a refactor.
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
from _shared import catalogue
from _shared.demo import fresh_store, show_cases, show_tree

from regression_records import (
    FunctionIdentity,
    ToleranceConfig,
    compare,
    enable,
    replay,
)

STORE = fresh_store(catalogue.STORE.root.name)

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# A shared ``RecordStore``
# ------------------------
#
# Passing a ``RecordStore`` instead of a path lets several decorations share one - and gives
# you the same object to query afterwards. ``tags`` attach free-form metadata to every
# record a decoration writes, which is how you keep track of what a batch of records came
# from. ``price`` lives in ``examples/_shared/catalogue.py``, beside the store it
# records to:

print(inspect.getsource(catalogue))

store = catalogue.STORE
price = catalogue.price

price(3, 1.5)
price(4, 2.0)
price(3, 2.5)

show_tree(STORE)

# %%
# Every record in the store
# -------------------------
#
# ``iter_record_paths()`` lists every record file under the store, sorted, by walking the
# directory tree - so a record that no longer loads is still listed. ``load_record()``
# reads one into a ``RegressionRecord`` without loading any side file. This pair is how
# ``replay`` itself walks a store.

for path in store.iter_record_paths():
    record = store.load_record(path)
    print(
        f"{path.relative_to(STORE)}\n"
        f"    {record.identity.module}:{record.identity.qualname}"
        f"  created {record.created_at}  tags={record.tags}"
    )

# %%
# The identity comes from the record, not from where the file sits: the directory is a
# filing convention, and a record stays what it says it is wherever it is copied.
#
# One function's records
# ----------------------
#
# The wrapper carries the identity its records are filed under, which is what the store's
# methods take.

identity = price.identity
print(identity)
print()

records = store.list_records(identity)
print(f"{len(records)} record(s):")
for record in records:
    print(
        f"  {record.record_id[:8]}  input_hash={record.input_hash[:12]}  tags={record.tags}"
    )

# %%
# Reading a recorded value back
# -----------------------------
#
# A record holds a JSON tree with references to side files; loading it is deliberately a
# separate step, so listing records or checking an identity does not pay for materializing
# every array they mention.
#
# ``store.deserialization_context(identity)`` supplies the directory those references
# resolve against.

ctx = store.deserialization_context(identity)

for record in records:
    args = record.load_args(ctx)
    kwargs = record.load_kwargs(ctx)
    output = record.load_output(ctx)
    print(
        f"price(*{args}, **{kwargs}) -> total={output['total']}, detail={output['detail']}"
    )

# %%
# The records of one call
# -----------------------
#
# ``input_hash`` groups the records of one call: the same function on the same
# arguments. It is the leading part of each record's filename, so those records sit
# side by side. A call normally has one record; it has more when its behavior was
# recorded, changed and recorded again (example 5), which is the thing to look for.

by_call = {}
for record in records:
    by_call.setdefault(record.input_hash, []).append(record.record_id[:8])
for input_hash, ids in by_call.items():
    print(f"  {input_hash[:12]}: {len(ids)} record(s) {ids}")
print("has any records:", store.has_records(identity))

# %%
# Comparing by hand
# -----------------
#
# ``compare()`` is a pure function, so a recorded value and a fresh one can be checked
# without going near replay. This is the seam to build a custom check on - a pytest
# helper, a report over a whole store, a comparison replay cannot make for you.

recorded = records[0].load_output(ctx)
fresh = {"total": 4.5001, "detail": np.full(3, 1.5)}

print(compare(recorded, fresh).summary())
print()
print(compare(recorded, fresh, ToleranceConfig(rtol=1e-3)).summary())

# %%
# When a function moves
# ---------------------
#
# Records are filed under ``module/qualname``, and replay imports a function by that
# recorded name. Moving a function to another module leaves its history pointing at the
# old name. This is not resolved automatically - which of several moved functions a set
# of records belongs to is not something a library can work out.
#
# ``rename_identity`` does it by hand:

moved_to = FunctionIdentity(
    module="pricing.engine",
    qualname="price",
    kind=identity.kind,
    code_hash=identity.code_hash,
    signature=identity.signature,
)

store.rename_identity(identity, moved_to)
show_tree(STORE)

# %%
# Each moved record's stored ``module`` and ``qualname`` are rewritten to match its new home,
# so the directory and the files inside it do not disagree. The rest of the recorded
# identity - ``kind``, ``code_hash``, ``signature`` - describes the code each record was
# *captured from* and is left alone, so moving a function does not erase the drift warning
# you would get if its implementation had also changed.
#
# Side files are not touched: records reference them relatively, so they move with the
# directory and keep resolving.

print("records still readable after the move:")
moved_ctx = store.deserialization_context(moved_to)
for record in store.list_records(moved_to):
    print(
        f"  {record.identity.module}.{record.identity.qualname}",
        record.load_output(moved_ctx)["total"],
    )

# %%
# ``rename_identity`` refuses to overwrite an existing destination, so merging two histories
# is something you have to do deliberately rather than by accident.
#
# Replay imports each record's function by its recorded name. There is no module
# ``pricing.engine`` in this example, so every moved record is ``not_importable`` - the
# status that, in a real move, says the rename has not been done yet (or was done to the
# wrong place).

report = replay(store)
show_cases(report)
print(report.failures[0].error)

# %%
# Removing a record, and pruning what it leaves
# ---------------------------------------------
#
# Records are never deleted for you. Removing one is a manual act - delete its file - and
# its side files stay behind in ``data/``. ``prune`` finds data files that no record
# names. It only reports by default:

moved_records = sorted(Path(STORE).rglob("pricing/engine/price/records/*.json"))
moved_records[0].unlink()

for path in store.prune(moved_to):
    print("unreferenced:", path.name)

# %%
# ``dry_run=False`` deletes them. With no identity, ``prune`` covers the whole store.

deleted = store.prune(dry_run=False)
print(f"deleted {len(deleted)} file(s); left to prune: {store.prune()}")

# %%
# Using it from pytest
# --------------------
#
# To check a store in a test suite, replay it. ``replay_tests`` (example 14) turns every
# record into its own pytest test:
#
# .. code-block:: python
#
#    # tests/test_regressions.py
#    from regression_records import replay_tests
#
#    test_regressions = replay_tests("tests/regression_data")
#
#
# Commit ``tests/regression_data/``. A pull request that changes behaviour then fails
# the case that moved, and one that re-records shows the changed records in its diff,
# which is the point: the reviewer sees *what* changed, not just that something did.

# %%
# Moving a store
# --------------
#
# Every reference inside a record is relative to that record's own directory, so a store -
# or a single identity directory within it - can be moved, copied, or archived as a unit
# and still resolve. That is what makes committing one to a repository work.
