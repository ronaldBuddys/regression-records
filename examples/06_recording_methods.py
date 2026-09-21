"""
record_method : recording and replaying methods
===============================================

A method's behavior is as much about the object it acts on as about its arguments, so
``record_method`` snapshots ``self`` **before** and **after** the call and records both.

That captures what ``record_func`` structurally cannot: a method that still returns the
right answer but stops updating its object correctly. Replay rebuilds ``self`` in its
recorded state, calls the method on the recorded arguments, and checks both the return
value and the state after.
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

from _shared import ledgers
from _shared.demo import fresh_store, show_cases, show_record

from regression_records import (
    RecordStore,
    SerializationError,
    enable,
    record_method,
    replay,
)

STORE = fresh_store(ledgers.STORE.name)

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# A recorded method
# -----------------
#
# ``Ledger`` lives in ``examples/_shared/ledgers.py``. Replay imports a method's class
# by name, as it does a function, so a class defined in the script itself - in
# ``__main__`` - is refused when its method is decorated.

print(inspect.getsource(ledgers.Ledger))

ledger = ledgers.Ledger(100.0)
print("returned:", ledger.apply(25.0))
print("state now:", ledger.balance, ledger.history)

# %%
# Both sides of the call are in the record
# ----------------------------------------
#
# ``state_before`` is deep-copied *before* the method runs. That copy is the whole
# point: a snapshot that shared the object's lists would quietly turn into a picture of
# the state afterwards, and the record would show nothing having changed.

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

ledgers.Ledger(999.0).apply(25.0)

store = RecordStore(STORE)
identity = ledgers.Ledger.apply.identity
ctx = store.deserialization_context(identity)
for record in store.list_records(identity):
    print(
        f"  input_hash {record.input_hash[:12]}  record {record.record_id[:8]}"
        f"  balance before {record.load_state_before(ctx)['balance']}"
        f"  returned {record.load_output(ctx)}"
    )

# %%
# Replaying them
# --------------
#
# For each record, replay imports ``Ledger``, rebuilds an instance from
# ``state_before``, calls the original ``apply`` on it with the recorded arguments,
# and compares the return value with ``output`` and the instance's state afterwards
# with ``state_after``. ``state_before`` is the input, so it is not compared.
#
# ``Ledger`` records its whole state, so replay rebuilds it the way unpickling does:
# ``Ledger.__new__(Ledger)``, then the recorded attributes set on it. ``__init__`` is
# not called - a constructor may need a database or an argument nobody kept - so an
# invariant that only ``__init__`` sets up, without storing it as an attribute, is not
# there on replay. A class with its own ``__getstate__`` / ``__setstate__`` is recorded
# and rebuilt through them instead, as ``pickle`` would.

show_cases(replay(STORE))

# %%
# The regression a return value hides
# -----------------------------------
#
# Suppose a later version of ``apply`` stops appending to ``history`` while keeping
# the arithmetic right. It still returns 125.0, so a test asserting on the return value
# passes. Switching off ``ledgers.KEEP_HISTORY`` stands in for that edit.

ledgers.KEEP_HISTORY = False
try:
    report = replay(STORE)
finally:
    ledgers.KEEP_HISTORY = True
show_cases(report)
print()
print(report.failures[0].describe())

# %%
# The return value matched; the difference is at ``$.state_after.history``, the
# regression ``record_func`` could not have seen.
#
# State that must not be recorded
# -------------------------------
#
# ``state_before`` is a deep copy, and plenty of real attributes cannot be deep-copied
# at all - locks, sockets, database connections, open files. Others can be copied but
# should not be compared, like a cache or a timestamp.
#
# By default every attribute is captured, so an object holding a connection fails
# loudly rather than recording something misleading:

try:
    ledgers.NaivePricer(ledgers.Connection()).value("ACME", 10)
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
# ``exclude_attrs`` names what to leave out; ``state_attrs`` names what to keep, from
# the other direction. Either way the record holds only part of the object, which is
# not enough to rebuild it: replay would have no connection to give ``value``. So both
# need ``factory=``, a zero-argument callable returning an instance, on which replay
# sets the recorded attributes. Leaving it out is refused when decorating:

try:
    record_method(STORE, exclude_attrs=("connection",))
except TypeError as e:
    print(f"TypeError: {e}")

# %%
# ``Pricer`` excludes its connection, and its factory opens a new one:

print(inspect.getsource(ledgers.new_pricer))
print(inspect.getsource(ledgers.Pricer))

pricer = ledgers.Pricer(ledgers.Connection())
print("returned:", pricer.value("ACME", 10))
show_record(
    STORE,
    keys=["state_before", "state_after"],
    select=lambda r: r["identity"]["qualname"] == "Pricer.value",
)

# %%
# ``state_attrs`` suits an object where only a couple of attributes matter.
# ``Survey`` records its origin and marks, custom objects that are recorded in full,
# recursively; its factory can be any callable, a lambda included, since replay calls
# it rather than importing it.

print(inspect.getsource(ledgers.Survey))

survey = ledgers.Survey()
survey.mark(4.0, 6.0)
print("origin:", survey.origin, "marks:", survey.marks)

# %%
# ``Point`` lives in ``_shared/models.py`` for the same reason the classes here live in
# ``_shared/ledgers.py``: a recorded object is re-imported by its class's module and
# name when the record is loaded (example 8).
#
# The decorator, not the record, decides how a receiver is rebuilt: replay uses the
# ``state_attrs``, ``exclude_attrs`` and ``factory`` the method is decorated with
# *now*. Every method record in the store replays:

show_cases(replay(STORE))

# %%
# A factory that returns an instance of another class, or a rebuild that raises, makes
# the case ``not_replayable``, with the reason in ``case.error``.
#
# The decorated method is a plain function assigned on the class, so nothing about
# ordinary Python is disturbed: descriptor binding, inheritance, ``isinstance``,
# ``functools.wraps`` metadata and ``super()`` all behave as usual.

print("__name__ preserved:  ", ledgers.Survey.mark.__name__)
print("binds normally:      ", type(ledgers.Survey().mark).__name__)
print("recorded identity:   ", identity.module, identity.qualname)
