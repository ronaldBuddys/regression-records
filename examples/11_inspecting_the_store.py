"""
Working with the store directly
===============================

A store is JSON in directories - ``grep``, ``git diff`` and ``jq`` all work on it. But there
is also an API, for the times you want to read a recorded value back as a Python object,
find out what a function's history actually covers, or move records after a refactor.
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
from _shared.demo import fresh_store, show_tree

from regression_records import (
    FunctionIdentity,
    RecordStore,
    ToleranceConfig,
    compare,
    record_func,
)

STORE = fresh_store("11_inspecting_the_store")

# %%
# A shared ``RecordStore``
# ------------------------
#
# Passing a ``RecordStore`` instead of a path lets several decorations share one - and gives
# you the same object to query afterwards. ``tags`` attach free-form metadata to every
# record a decoration writes, which is how you keep track of what a batch of records came
# from.

store = RecordStore(STORE)


@record_func(store, tags={"suite": "pricing", "owner": "risk-team"})
def price(quantity, unit):
    return {"total": quantity * unit, "detail": np.full(quantity, unit, dtype=float)}


price(3, 1.5)
price(4, 2.0)
price(3, 2.5)

show_tree(STORE)

# %%
# What has been recorded
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
# Finding one case
# ----------------
#
# ``find_by_input_hash`` is the lookup the decorator itself uses. The hash is the leading
# part of each record's filename, so this is a directory glob rather than a scan - it stays
# cheap as a function accumulates hundreds of recorded cases.

target = records[0].input_hash
found = store.find_by_input_hash(identity, target)
print("found:", found.record_id[:8], "| created:", found.created_at)
print("has any records:", store.has_records(identity))

# %%
# Comparing by hand
# -----------------
#
# ``compare()`` is a pure function, so a recorded value and a fresh one can be checked
# without going near a decorator. This is the seam to build a custom assertion on - a
# pytest helper, a report over a whole store, a check that runs somewhere the decorator
# cannot.

recorded = store.find_by_input_hash(identity, target).load_output(ctx)
fresh = {"total": 4.5001, "detail": np.full(3, 1.5)}

print(compare(recorded, fresh).summary())
print()
print(compare(recorded, fresh, ToleranceConfig(rtol=1e-3)).summary())

# %%
# When a function moves
# ---------------------
#
# Records are filed under ``module/Qualname``, so moving a function to another module leaves
# its history behind under the old path, and the next validating run reports
# ``RecordNotFoundError``. This is not resolved automatically - which of several moved
# functions a set of records belongs to is not something a library can work out.
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
# There is no pytest plugin yet. The decorator is enough on its own - decorate the
# function under test and the assertions are the recorded history:
#
# .. code-block:: python
#
#    # conftest.py
#    import os
#    from regression_records import RunMode
#
#    STORE = "tests/regression_data"
#    MODE = RunMode.VALIDATE_ONLY if os.environ.get("CI") else RunMode.BOTH
#
#
# .. code-block:: python
#
#    # test_pricing.py
#    from mypkg.pricing import price          # decorated at its definition
#
#    def test_price_regressions():
#        price(3, 1.5)                        # raises RegressionMismatchError on a change
#        price(4, 2.0)
#
#
# Commit ``tests/regression_data/``. A pull request that changes behaviour then shows the
# changed records in its diff, which is the point: the reviewer sees *what* changed, not
# just that something did.
#
# To accept a change, re-run once with ``overwrite=OverwritePolicy.IF_MISMATCH`` (example 5)
# and commit the updated records.

# %%
# Moving a store
# --------------
#
# Every reference inside a record is relative to that record's own directory, so a store -
# or a single identity directory within it - can be moved, copied, or archived as a unit
# and still resolve. That is what makes committing one to a repository work.
