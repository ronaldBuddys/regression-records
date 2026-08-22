# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 6. Recording methods
#
# A method's behaviour is as much about the object it acts on as about its arguments, so
# `record_method` snapshots `self` **before** and **after** the call and records both.
#
# That catches the regression `record_func` structurally cannot see: a method that still
# returns the right answer but stops updating its object correctly.

# %%
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd()))

from _shared.demo import fresh_store, show_record, show_result
from _shared.models import Point
from regression_records import RegressionMismatchError, SerializationError, record_method

STORE = fresh_store("06_recording_methods")

KEEP_HISTORY = True


class Ledger:
    def __init__(self, balance):
        self.balance = balance
        self.history = []

    @record_method(STORE)
    def apply(self, delta):
        if KEEP_HISTORY:
            self.history.append(self.balance)
        self.balance += delta
        return self.balance


ledger = Ledger(100.0)
print("returned:", ledger.apply(25.0))
print("state now:", ledger.balance, ledger.history)

# %% [markdown]
# ## Both sides of the call are in the record
#
# `state_before` is deep-copied *before* the method runs. That copy is the whole point: a
# snapshot that shared the object's lists would quietly turn into a picture of the state
# afterwards, and the record would show nothing having changed.

# %%
show_record(STORE, keys=["invocation", "state_before", "state_after", "output"])

# %% [markdown]
# Notice `invocation.args` is empty and `self` appears nowhere. The receiver is described
# by its captured state instead - hashing the instance as an argument would make every
# instance, and (since most objects hash by identity) every run, a separate case.
#
# The consequence is worth stating plainly: **two instances in different states, called
# with the same arguments, are one case.** The difference in state is reported as a
# mismatch rather than recorded as a second case. If the starting state is what
# distinguishes two calls, pass it as an argument instead.

# %%
other = Ledger(999.0)  # same call, different starting state
try:
    other.apply(25.0)
except RegressionMismatchError as e:
    show_result(e.result)

# %% [markdown]
# ## The regression a return value hides
#
# Break the bookkeeping while leaving the arithmetic correct. `apply` still returns 125.0,
# so a test asserting on the return value passes - and the record catches it anyway,
# because `$.state_after.history` is part of the recording.

# %%
KEEP_HISTORY = False

try:
    Ledger(100.0).apply(25.0)
except RegressionMismatchError as e:
    show_result(e.result)

KEEP_HISTORY = True

# %% [markdown]
# ## State that must not be recorded
#
# `state_before` is a deep copy, and plenty of real attributes cannot be deep-copied at
# all - locks, sockets, database connections, open files. Others can be copied but should
# not be compared, like a cache or a timestamp.
#
# By default every attribute is captured, so such an object fails loudly rather than
# recording something misleading:

# %%
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

# %% [markdown]
# `exclude_attrs` names what to leave out; `state_attrs` names what to keep, from the
# other direction. Use `exclude_attrs` when the object is mostly interesting, and
# `state_attrs` when only a couple of attributes matter.

# %%
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
    next((STORE / "__main__" / "WorkingService.handle" / "records").glob("*.json")).read_text()
)
print("recorded state_before:", recorded["state_before"])
print("recorded state_after: ", recorded["state_after"])

# %% [markdown]
# ## Nested objects in state
#
# Captured state is serialized the same way anything else is, so an attribute holding a
# custom object, an array, or a dataframe is recorded in full - recursively.

# %%
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

# %% [markdown]
# `Point` lives in `_shared/models.py` rather than in this file on purpose. A class is
# recorded as its module and qualified name so it can be re-imported when the record is
# loaded, and a class defined in `__main__` - which is what a script or a notebook cell
# is - cannot be. Example 8 covers that in full.
#
# The decorated method is a plain function assigned on the class, so nothing about
# ordinary Python is disturbed: descriptor binding, inheritance, `isinstance`,
# `functools.wraps` metadata and `super()` all behave as usual.

# %%
print("__name__ preserved:  ", Survey.mark.__name__)
print("binds normally:      ", type(Survey().mark).__name__)
print("recorded identity:   ", Survey.mark.identity.module, Survey.mark.identity.qualname)
