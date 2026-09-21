"""
record_method : recording methods and object state
==================================================

A method's behavior is as much about the object it acts on as about its arguments, so
``record_method`` snapshots ``self`` **before** and **after** the call and records both.

That captures what ``record_func`` structurally cannot: a method that still returns the
right answer but stops updating its object correctly.

Method records are recorded and stored in full, but this release does not replay them
yet: replay reports them as ``not_replayable``. The next release rebuilds ``self`` from
the recorded state and checks both the return value and the state after.
"""

import json
import sys
import threading
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared.demo import fresh_store, show_record, show_result
from _shared.models import Point

from regression_records import (
    RecordStore,
    SerializationError,
    compare,
    enable,
    record_method,
)

STORE = fresh_store("06_recording_methods")

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# A recorded method
# -----------------
#
# ``Ledger`` is defined in this script, which is fine for recording. (Replay will need
# it importable by name, as it does a function.)


class Ledger:
    def __init__(self, balance):
        self.balance = balance
        self.history = []

    @record_method(STORE)
    def apply(self, delta):
        self.history.append(self.balance)
        self.balance += delta
        return self.balance


ledger = Ledger(100.0)
print("returned:", ledger.apply(25.0))
print("state now:", ledger.balance, ledger.history)

# %%
# Both sides of the call are in the record
# ----------------------------------------
#
# ``state_before`` is deep-copied *before* the method runs. That copy is the whole point: a
# snapshot that shared the object's lists would quietly turn into a picture of the state
# afterwards, and the record would show nothing having changed.

show_record(STORE, keys=["invocation", "state_before", "state_after", "output"])

# %%
# Notice ``invocation.args`` is empty and ``self`` appears nowhere. The receiver is
# described by its captured state instead: hashing the instance as an argument would
# make every instance, and (since most objects hash by identity) every run, a separate
# case.
#
# Two instances in different states
# ---------------------------------
#
# The same call on an object in another state is another record. Both records share the
# call's input hash, which covers the arguments only; the state before tells them apart.

Ledger(999.0).apply(25.0)

store = RecordStore(STORE)
ctx = store.deserialization_context(Ledger.apply.identity)
records = store.list_records(Ledger.apply.identity)
for record in records:
    print(
        f"  input_hash {record.input_hash[:12]}  record {record.record_id[:8]}"
        f"  balance before {record.load_state_before(ctx)['balance']}"
        f"  returned {record.load_output(ctx)}"
    )

# %%
# The regression a return value hides
# -----------------------------------
#
# Suppose a later version of ``apply`` stops appending to ``history`` while keeping the
# arithmetic right. It still returns 125.0, so a test asserting on the return value
# passes. The record holds ``state_after`` too, and that is where the change shows.
# Method replay will make this comparison for you; here it is by hand, with
# ``compare()``, between the recorded state after and what the broken version leaves.

first = next(r for r in records if r.load_state_before(ctx)["balance"] == 100.0)
recorded_after = first.load_state_after(ctx)
broken_after = {"balance": 125.0, "history": []}  # no longer appended to

show_result(compare(recorded_after, broken_after, root="$.state_after"))

# %%
# State that must not be recorded
# -------------------------------
#
# ``state_before`` is a deep copy, and plenty of real attributes cannot be deep-copied at
# all - locks, sockets, database connections, open files. Others can be copied but should
# not be compared, like a cache or a timestamp.
#
# By default every attribute is captured, so such an object fails loudly rather than
# recording something misleading:


class Service:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    @record_method(STORE)
    def handle(self, request):
        self.calls += 1
        return request.upper()


try:
    Service().handle("ping")
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
# ``exclude_attrs`` names what to leave out; ``state_attrs`` names what to keep, from the
# other direction. Use ``exclude_attrs`` when the object is mostly interesting, and
# ``state_attrs`` when only a couple of attributes matter.


class WorkingService:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    @record_method(STORE, exclude_attrs=("lock",))
    def handle(self, request):
        self.calls += 1
        return request.upper()


print("returned:", WorkingService().handle("ping"))

recorded = json.loads(
    next(
        (STORE / "__main__" / "WorkingService.handle" / "records").glob("*.json")
    ).read_text()
)
print("recorded state_before:", recorded["state_before"])
print("recorded state_after: ", recorded["state_after"])

# %%
# Nested objects in state
# -----------------------
#
# Captured state is serialized the same way anything else is, so an attribute holding a
# custom object, an array, or a dataframe is recorded in full - recursively.


class Survey:
    def __init__(self):
        self.origin = Point(0.0, 0.0)
        self.marks = []

    @record_method(STORE, state_attrs=("origin", "marks"))
    def mark(self, x, y):
        self.marks.append(Point(x, y))
        self.origin = Point(x / 2, y / 2)
        return len(self.marks)


survey = Survey()
survey.mark(4.0, 6.0)
print("origin:", survey.origin, "marks:", survey.marks)

# %%
# ``Point`` lives in ``_shared/models.py`` rather than in this file on purpose. A class is
# recorded as its module and qualified name so it can be re-imported when the record is
# loaded, and a class defined in ``__main__`` - which is what a script or a notebook cell
# is - cannot be. Example 8 covers that in full.
#
# The decorated method is a plain function assigned on the class, so nothing about
# ordinary Python is disturbed: descriptor binding, inheritance, ``isinstance``,
# ``functools.wraps`` metadata and ``super()`` all behave as usual.

print("__name__ preserved:  ", Survey.mark.__name__)
print("binds normally:      ", type(Survey().mark).__name__)
print(
    "recorded identity:   ", Survey.mark.identity.module, Survey.mark.identity.qualname
)

# %%
# Replaying method records
# ------------------------
#
# Not in this release: ``replay`` reports every method record as ``not_replayable``,
# and ``replay_tests`` fails them rather than skipping them, so a store of method
# records is not quietly treated as checked. The next release replays them: it rebuilds
# ``self`` from ``state_before`` (or with a ``factory=`` when ``state_attrs`` recorded
# only part of the object), calls the method on the recorded arguments, and compares
# the return value and ``state_after``, just as the comparison above did by hand.
