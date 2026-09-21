"""
Gotchas and edge cases
======================

The behaviour that surprises people, run rather than described. Most of it follows from
two rules: **a call is matched by an exact hash of its arguments**, and **a call with no
match is, by default, recorded rather than failed**. The first group below matters most -
each is a way for a suite to be green without having tested anything.
"""

import contextlib
import sys
import warnings
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from collections import namedtuple

import numpy as np
from _shared.demo import fresh_store, record_paths, show_result
from _shared.models import Point, Quote

from regression_records import (
    CodeDriftWarning,
    RecordNotFoundError,
    RegressionMismatchError,
    RunMode,
    SerializationError,
    ToleranceConfig,
    compare,
    record_func,
    record_method,
)

STORE = fresh_store("13_gotchas")


def scenario(name):
    """A separate store per section, so the sections do not see each other's records."""
    return STORE / name


# %%
# An empty store passes
# ---------------------
#
# Under the default ``RunMode.BOTH`` a call with no history is recorded, and recording
# cannot fail. So if the store is missing - not committed, git-ignored, wiped by a
# clean-up step - every call is "new" and the suite is green.


@record_func(scenario("empty"))
def total(values):
    return sum(values)


total([1, 2, 3])
print("first run against an empty store:", total.last_result, "- nothing was compared")

# %%
# ``RunMode.VALIDATE_ONLY`` is the cure, and is what CI should run: no history is an
# error there. Options can be given as strings; a misspelt one fails when the decorator
# is applied, not on some later call.

strict = record_func(scenario("never-recorded"), mode="validate_only")(
    total.__wrapped__
)
try:
    strict([1, 2, 3])
except RecordNotFoundError as e:
    print("RecordNotFoundError:", str(e)[:60], "...")

try:
    record_func(STORE, mode="validate-only")
except ValueError as e:
    print("ValueError:", e)

# %%
# Changing a signature orphans the history
# ----------------------------------------
#
# Arguments are bound to the signature before hashing, defaults included - that is what
# makes ``f(1, 2)`` and ``f(1, b=2)`` one case. The cost: add a parameter, even one with
# a default, and *every* call hashes differently. Nothing matches, everything is
# recorded afresh, and the old records sit unreachable without a word.


def scale_v1(x):
    return x * 2


def scale_v2(x, factor=2):
    return x * factor + 1  # a bug rides in with the new parameter


record_func(scenario("signature"), name="scale")(scale_v1)(10)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", CodeDriftWarning)
    after = record_func(scenario("signature"), name="scale")(scale_v2)
    after(10)

print("result changed 20 -> 21, comparison ran:", after.last_result is not None)
print("records now:", len(record_paths(scenario("signature"))), "(one is orphaned)")

# %%
# After a signature change, re-record deliberately and review the store's diff.
#
# ``RECORD_ONLY`` does not re-record
# ----------------------------------
#
# With the default ``OverwritePolicy.NEVER``, history is immutable - so ``RECORD_ONLY``
# meeting a case it already has does nothing at all: no write, no comparison.

drift = {"by": 0}


def reading(x):
    return x + drift["by"]


record_func(scenario("record-only"))(reading)(1)
drift["by"] = 100
recording = record_func(scenario("record-only"), mode=RunMode.RECORD_ONLY)(reading)
print("returned", recording(1), "- compared:", recording.last_result is not None)

# %%
# Methods are matched by arguments, not by the object
# ---------------------------------------------------
#
# ``self`` is not hashed - its state is recorded instead. So two objects in different
# states making the same call are the *same case*: the second is compared against the
# first's state and fails, rather than becoming a case of its own.


class Account:
    def __init__(self, balance):
        self.balance = balance

    @record_method(scenario("instances"))
    def deposit(self, amount):
        self.balance += amount
        return self.balance


Account(100).deposit(10)
try:
    Account(500).deposit(10)
except RegressionMismatchError as e:
    show_result(e.result)

# %%
# If the starting state is part of what distinguishes your cases, pass it as an argument
# (or record a plain function that builds the object and calls the method).
#
# ``state_attrs`` selects attributes, not properties
# --------------------------------------------------
#
# A name that matches nothing - a typo, or a property, which lives on the class rather
# than in the instance - is an error, because the method would otherwise be tested
# against an empty state and pass.


class Basket:
    def __init__(self):
        self.items = []

    @property
    def size(self):
        return len(self.items)

    @record_method(scenario("state-attrs"), state_attrs=["size"])
    def add(self, item):
        self.items.append(item)


try:
    Basket().add("apple")
except SerializationError as e:
    print("SerializationError:", str(e)[:88], "...")

# %%
# Integers are compared within tolerance
# --------------------------------------
#
# ``bool`` is exempt from tolerance; ``int`` is not. With the default ``rtol=1e-7`` a
# large count can move and still pass. Give counts and identifiers an exact override.

print("default:", compare(1_000_000_000, 1_000_000_050).passed)
print("exact:  ", compare(1_000_000_000, 1_000_000_050, ToleranceConfig(rtol=0)).passed)

# %%
# NaN, null and signed zero
# -------------------------
#
# ``NaN`` equals ``NaN`` by default (``equal_nan=True``), because a recorded NaN usually
# means "no value here" and should keep meaning it. As an *argument*, NaN matches
# itself too. Signed zeros compare equal as results but hash differently as arguments,
# so they are two cases.

print("NaN result vs NaN:", compare(float("nan"), float("nan")).passed)


@record_func(scenario("zero"))
def identity(x):
    return x


identity(0.0)
identity(-0.0)
print("cases recorded for 0.0 and -0.0:", len(record_paths(scenario("zero"))))

# %%
# A polars ``null`` is not a NaN, though: one is a missing value and the other a number,
# and a column that changed from one to the other is reported whatever ``equal_nan``
# says. pandas has a single missing value per dtype, so the question does not arise.

try:
    import polars as pl

    nulls = pl.DataFrame({"x": [1.0, None]})
    nans = pl.DataFrame({"x": [1.0, float("nan")]})
    print("null vs NaN:", compare(nulls, nans).passed)
except ImportError:
    print("polars is not installed")

# %%
# A frame argument is matched by its bytes
# ----------------------------------------
#
# Arguments are matched by hashing their serialized form, and for a dataframe that
# includes the name of its parquet file - which is the hash of the file's bytes. Equal
# frames give equal bytes from run to run, so this works. But the bytes are the
# encoder's to choose: polars changed them between 1.30 and 1.40, for one. After such an
# upgrade, a call that takes a frame no longer finds its record, and under the default
# mode is quietly recorded as a new case instead of being validated. Upgrade the
# library in a commit of its own, run with ``on_new_case="error"`` to see it, and
# re-record.

try:
    import io

    import polars as pl

    from regression_records.serialization.naming import content_filename

    buffer = io.BytesIO()
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(buffer)
    print(
        f"polars {pl.__version__}:",
        content_filename("df", buffer.getvalue(), extension="parquet"),
    )
except ImportError:
    print("polars is not installed")

# %%
# What cannot be recorded
# -----------------------
#
# Each of these is refused when recording, with an error that says what to do - rather
# than writing a record that silently lost something or could never be read back.

Local = namedtuple("Local", "bid ask")  # defined in __main__: cannot be re-imported

parent = {"name": "parent", "children": []}
parent["children"].append({"name": "child", "parent": parent})  # refers back to itself

unrecordable = {
    "a masked array": np.ma.masked_array([1.0, 2.0], mask=[False, True]),
    "a self-referencing structure": parent,
    "a class from __main__": Local(1.0, 1.1),
    "an object array": np.array([1, "two"], dtype=object),
    "a dict with non-string keys": {1: "one"},
}

for label, value in unrecordable.items():
    try:
        record_func(scenario("refused"), name="emit")(lambda value=value: value)()
    except SerializationError as e:
        print(f"{label}:\n    {str(e)[:96]}...")

# %%
# The same types from an importable module are fine - a named tuple keeps its class, and
# comes back as itself:


@record_func(scenario("named-tuple"))
def quote():
    return Quote(bid=99.5, ask=100.5), Point(1.0, 2.0)


quote()
quote()
print("named tuple and object validated:", quote.last_result.passed)

# %%
# Generator and async functions are refused earlier still, when decorated: calling one
# returns a generator or coroutine, which is not a result.

try:

    @record_func(STORE)
    def stream():
        yield 1

except TypeError as e:
    print("TypeError:", str(e)[:82], "...")

# %%
# Only the exception's type is compared
# -------------------------------------
#
# A recorded exception is matched on its class name. Messages routinely contain values,
# paths and addresses, so a changed message is not a regression - but neither is a
# different exception that happens to share the name.

message = {"text": "first wording"}


@record_func(scenario("exceptions"))
def fails():
    raise ValueError(message["text"])


for text in ("first wording", "completely different wording"):
    message["text"] = text
    with contextlib.suppress(ValueError):
        fails()
print("passed despite the new message:", fails.last_result.passed)

# %%
# A nested regression surfaces as itself
# --------------------------------------
#
# When one recorded function calls another and the inner one regresses, the inner
# ``RegressionMismatchError`` propagates straight out. It is not recorded as "what the
# outer function raises".

offset = {"by": 0}


@record_func(scenario("nested"))
def inner(x):
    return x + offset["by"]


@record_func(scenario("nested"))
def outer(x):
    return inner(x) * 2


outer(1)
offset["by"] = 5
try:
    outer(1)
except RegressionMismatchError as e:
    print(
        "raised from:", e.result.diffs[0].path, "of inner -", e.result.diffs[0].message
    )

# %%
# Things this page cannot show
# ----------------------------
#
# * **Parallel test runs.** Writes are atomic, so validating under ``pytest -n auto`` is
#   fine, but two workers *recording* the same new case each write a record. Record from
#   one process.
# * **Trust.** Loading a record imports the module a recorded object names. Review
#   changes to a store as you review the code beside it, and see the Trust section of the
#   persistence page.
# * **Pickle is opt-in.** A pandas frame that parquet cannot hold is refused unless you
#   ask for it to be pickled - example 10 shows how - and a pickle is never read without
#   the same opt-in.
# * **Drift warnings.** Editing a recorded function's body emits ``CodeDriftWarning`` on
#   the next validation (example 7). Changing only its decorator's arguments, its
#   comments or its layout does not.
