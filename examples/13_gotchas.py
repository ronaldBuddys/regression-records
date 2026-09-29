"""
``replay`` : gotchas, what a replay cannot see
==============================================

The behavior that surprises people, run rather than described. Most of it follows from
two rules: **a call is replayed from its recorded inputs alone**, and **replay compares
with what was recorded, not with what is right**. The first section matters most: it
is how a suite can be green without having tested anything.

The recorded functions live in ``examples/_shared/gotchas.py``, where replay can import
them.
"""

import functools
import inspect
import sys
from datetime import date, timedelta
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from collections import namedtuple

import numpy as np
from _shared import gotchas
from _shared.demo import fresh_store, show_cases, show_counts

from regression_records import (
    RecordStore,
    SerializationError,
    ToleranceConfig,
    compare,
    enable,
    record_func,
    recording,
    replay,
)

STORE = fresh_store(gotchas.STORE.name)
gotchas.AUDIT_LOG.unlink(missing_ok=True)


def records_of(recorded):
    """Return the records of one recorded function in ``_shared.gotchas``."""
    return RecordStore(STORE).list_records(recorded.identity)


def replay_only(qualname, **replay_kwargs):
    """Replay the records of one function in ``_shared.gotchas``."""
    return replay(STORE, select=f"_shared.gotchas:{qualname}", **replay_kwargs)


# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# A run without the switch records nothing
# ----------------------------------------
#
# Recording is off unless ``REGRESSION_RECORDS`` switches it on. A run meant to record
# that was started without it writes nothing, and says nothing about it.

with recording(False):
    gotchas.average([1.0, 2.0])
show_counts(STORE)

# %%
# An empty store passes
# ---------------------
#
# A store that exists but holds no records - never recorded, wiped, or git-ignored so CI
# checks out an empty directory - replays nothing, and passes. Check the number of cases
# where that matters. A path that does not exist at all is an error.

report = replay(STORE)
assert report.passed and len(report.cases) == 0, "an empty store replays nothing"

try:
    replay(STORE / "no-such-store")
except FileNotFoundError as e:
    print("FileNotFoundError:", e)

# %%
# A function that reads a clock
# -----------------------------
#
# ``days_to_expiry`` reads today's date. Replay calls it with the recorded expiry, but
# the date it reads is the day of the replay, not the day of the recording.

print(inspect.getsource(gotchas.days_to_expiry))
print("recorded:", gotchas.days_to_expiry(date(2030, 1, 1)))

# %%
# Moving the clock on a day stands in for replaying tomorrow:

original_today = gotchas.today
gotchas.today = lambda: date.today() + timedelta(days=1)
try:
    report = replay_only("days_to_expiry")
finally:
    gotchas.today = original_today
print(report.failures[0].describe())

# %%
# That is a limit of replay, not a regression. Pass the date in as an argument, and the
# record holds it.
#
# A function that reads a global
# ------------------------------
#
# ``to_usd`` reads rates loaded at start-up. The replaying process has whatever rates it
# loaded, and if the one the record needs is missing, the call raises - reported as
# ``raised``, the status for a record that returned but raises on replay.

print("recorded:", gotchas.to_usd(100.0, "GBP"))

saved_rates = dict(gotchas.FX_RATES)
gotchas.FX_RATES.pop("GBP")  # a process that loaded only EUR
try:
    report = replay_only("to_usd")
finally:
    gotchas.FX_RATES.update(saved_rates)
show_cases(report)
print(report.failures[0].error)

# %%
# Side effects happen again
# -------------------------
#
# Replay really calls the function. ``log_trade`` appends to a log file, so replaying it
# appends again, as a unit test calling it would.

gotchas.log_trade("T-1")
replay_only("log_trade")
print(gotchas.AUDIT_LOG.read_text())

# %%
# A change upstream of the arguments is invisible
# -----------------------------------------------
#
# ``average`` is recorded on what ``load_prices`` returned. Replay passes it those
# recorded prices, so a broken ``load_prices`` does not reach ``average``'s records.
# Record at each stage of a pipeline you want pinned.

gotchas.average(gotchas.load_prices())

original_load = gotchas.load_prices
gotchas.load_prices = lambda: [0.0, 0.0, 0.0]  # broken
try:
    report = replay_only("average")
finally:
    gotchas.load_prices = original_load
show_cases(report)

# %%
# No code drift either: drift tracks the recorded function's own source, not the code
# it depends on, and ``average`` did not change.
#
# An argument that does not load back as itself
# ---------------------------------------------
#
# Replay calls the function with the recorded arguments loaded back from the record,
# so an argument must round-trip exactly. Some types do not: an ``IntEnum`` member is
# recorded as its integer (``docs/limitations.md`` lists the others). ``label`` reads
# ``side.name``, so on replay it receives ``1`` and raises.

print(inspect.getsource(gotchas.label))
print("recorded:", gotchas.label(gotchas.Side.BUY))

report = replay_only("label")
show_cases(report)
print(report.failures[0].error)

# %%
# A rebuilt object skips ``__init__``
# -----------------------------------
#
# A method record holds the object's state, and replay rebuilds the object from it
# without calling ``__init__``, as unpickling does. Anything ``__init__`` does besides
# setting attributes does not happen. ``Account.__init__`` registers the account in a
# module-level set, which ``close`` then removes it from:

print(inspect.getsource(gotchas.Account))
print("recorded:", gotchas.Account("A-1").close())

report = replay_only("Account.close")
show_cases(report)
print(report.failures[0].error)

# %%
# The rebuilt account was never registered, so ``close`` raises. Keep what a method
# relies on in the object's own attributes, or rebuild it with ``factory=`` (example 6).
#
# Functions replay could not find
# -------------------------------
#
# Replay imports a function by its recorded module and qualified name. A function that
# has no such name is refused when it is decorated, rather than recorded into a store
# that no replay can check. A lambda has no name, and a function defined inside
# another has ``<locals>`` in its qualified name:

try:
    record_func(STORE)(lambda x: 2 * x)
except TypeError as e:
    print("TypeError:", e)


def make_adder(n):
    @record_func(STORE)
    def add(x):
        return x + n

    return add


try:
    make_adder(1)
except TypeError as e:
    print("TypeError:", e)

# %%
# A ``functools.partial`` has no qualified name at all. Decorate the function it wraps.

try:
    record_func(STORE)(functools.partial(round, ndigits=2))
except TypeError as e:
    print("TypeError:", e)

# %%
# A function defined in a script belongs to ``__main__``. Replayed from the process
# that ran the script it might even be found, which would make this easy to miss; from
# any other process - pytest, CI - ``__main__`` is something else, and it is not. So it
# is refused too, and so is a method of a class defined in the script.

try:

    @record_func(STORE)
    def triple(x):
        return 3 * x

except TypeError as e:
    print("TypeError:", e)

# %%
# Earlier versions recorded such functions without complaint. Their records may still
# sit in a store, and they replay as ``not_importable``.
#
# A name that still imports but is no longer decorated is ``not_replayable``: replay
# only calls a function beneath a recording decorator. Taking the decorator off
# ``average`` shows it.

decorated_average = gotchas.average
gotchas.average = gotchas.average.__wrapped__  # as if the decorator were removed
try:
    report = replay_only("average")
finally:
    gotchas.average = decorated_average
show_cases(report)
print(report.cases[0].error)

# %%
# Only the exception's type is compared
# -------------------------------------
#
# A recorded exception is matched on its type. Messages routinely contain values, paths
# and addresses, so a changed message is not a regression.

try:
    gotchas.fail()
except ValueError as e:
    print("recorded:", e)

gotchas.MESSAGE = "completely different wording"
try:
    report = replay_only("fail")
finally:
    gotchas.MESSAGE = "first wording"
show_cases(report)

# %%
# Integers are compared exactly
# -----------------------------
#
# Counts and identifiers are not measurements, so no tolerance applies to them - not
# even a generous one. Tolerance applies once a float is involved: a count that is now
# computed as a float is compared within it.

loose = ToleranceConfig(rtol=1e-3)
assert not compare(1_000_000_000, 1_000_000_050, loose).passed, "ints are exact"
assert compare(1_000_000_000, 1_000_000_050.0, loose).passed, "a float is not"

# %%
# NaN, null and signed zero
# -------------------------
#
# ``NaN`` equals ``NaN`` by default (``equal_nan=True``), because a recorded NaN usually
# means "no value here" and should keep meaning it. Signed zeros compare equal as
# results but hash differently as arguments, so recording both gives two records.

assert compare(float("nan"), float("nan")).passed, "NaN result vs NaN"
assert compare(0.0, -0.0).passed, "0.0 result vs -0.0"

gotchas.echo(0.0)
gotchas.echo(-0.0)
print("records for 0.0 and -0.0:", len(records_of(gotchas.echo)))

# %%
# A polars ``null`` is not a NaN, though: one is a missing value and the other a number,
# and a column that changed from one to the other is reported whatever ``equal_nan``
# says. pandas has a single missing value per dtype, so the question does not arise.

try:
    import polars as pl

    nulls = pl.DataFrame({"x": [1.0, None]})
    nans = pl.DataFrame({"x": [1.0, float("nan")]})
    assert not compare(nulls, nans).passed, "a polars null is not a NaN"
except ImportError:
    print("polars is not installed")

# %%
# A frame argument is hashed by its bytes
# ---------------------------------------
#
# A call's input hash covers its serialized arguments, and for a dataframe that
# includes the name of its parquet file - the hash of the file's bytes. Equal frames
# give equal bytes from run to run, but the bytes are the encoder's to choose: polars
# changed them between 1.30 and 1.40, for one. After such an upgrade, re-recording a
# call that takes a frame adds a record beside the old one, rather than converging with
# it. Both still replay, since nothing is looked up by input hash; delete whichever you
# no longer want.

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
# A call is recorded once per process
# -----------------------------------
#
# Within one process a call is recorded once, keyed on its arguments, not its result.
# A function that returns something different the second time it is called with the
# same arguments has only its first result recorded. Re-record in a fresh process
# (example 5).

print("returned:", gotchas.tick(1), gotchas.tick(1), gotchas.tick(1))
print("records:", len(records_of(gotchas.tick)))

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

for description, value in unrecordable.items():
    try:
        gotchas.echo(value)
    except SerializationError as e:
        print(f"{description}:\n    {str(e)[:96]}...")

# %%
# The same types from an importable module are fine - a named tuple keeps its class,
# and comes back as itself:

gotchas.quote_and_point()
show_cases(replay_only("quote_and_point"))

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
# ``state_attrs`` selects attributes, not properties
# --------------------------------------------------
#
# A name that matches nothing - a typo, or a property, which lives on the class rather
# than in the instance - is an error, because the method would otherwise be recorded
# with an empty state.

print(inspect.getsource(gotchas.Basket))

try:
    gotchas.Basket().add("apple")
except SerializationError as e:
    print("SerializationError:", str(e)[:88], "...")

# %%
# Things this page cannot show
# ----------------------------
#
# * **Parallel recording.** Writes are atomic, and several processes may record to one
#   store at once, as under ``pytest -n auto``: identical recordings converge on one
#   file, and changed ones add records. Do not run ``RecordStore.prune`` while anything
#   is recording.
# * **Trust.** Loading a record imports the module a recorded object names. Review
#   changes to a store as you review the code beside it, and see the Trust section of the
#   persistence page.
# * **Pickle is opt-in.** A pandas frame that parquet cannot hold is refused unless you
#   ask for it to be pickled - example 10 shows how - and a pickle is never read without
#   the same opt-in.
# * **Drift warnings.** Under ``replay_tests``, a case that passes although its
#   function's source changed since recording warns with ``CodeDriftWarning``
#   (example 7). Changing only its decorator's arguments, its comments or its layout
#   does not.
