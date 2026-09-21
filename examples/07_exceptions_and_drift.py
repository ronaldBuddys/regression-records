"""
Exceptions and code drift : what the replay report shows
========================================================

Raising is behaviour too. A function that used to reject bad input and now silently
accepts it has regressed just as surely as one that returns the wrong number, so
exceptions are recorded, and replay compares them alongside return values.

This example records a parser that raises on bad input, then changes its behaviour in
each of the ways the replay report distinguishes. It ends with code drift: the
function's source having changed since it was recorded, which replay reports as a
column on each case rather than as a failure.
"""

import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared import ports
from _shared.demo import fresh_store, record_paths, show_cases, show_record

from regression_records import record_func, replay

STORE = fresh_store(ports.STORE.name)

print(inspect.getsource(ports.parse_port))

# %%
# Recording a call that raises
# ----------------------------
#
# The exception propagates as usual - recording never swallows or substitutes what a
# function does - and it is recorded as the call's result.

print("ok:", ports.parse_port("8080"))

try:
    ports.parse_port("99999")
except ValueError as e:
    print("raised and recorded:", e)

# %%
# What a recorded exception holds
# -------------------------------
#
# Exactly one of ``output`` / ``exception`` is set on a record. The exception's type,
# message and formatted traceback are all stored - but only the **type** takes part in
# the comparison. Messages routinely carry paths, addresses and timings that would turn
# every run into a false mismatch; the message and traceback are there for you to read.

show_record(STORE, keys=["exception"], select=lambda r: "exception" in r)

# %%
# Replaying it
# ------------
#
# On replay the parser raises ``ValueError`` again, which is what the record says it
# did, so that case passes just like the one that returned.

show_cases(replay(STORE))

# %%
# ``replay_changed`` below replays the store with one of the parser's settings changed,
# and restores it afterwards whatever happens. Each setting stands in for an edit to
# ``ports.py``.


@contextmanager
def changed(**settings):
    """Temporarily set module-level settings of ``_shared.ports``."""
    saved = {name: getattr(ports, name) for name in settings}
    for name, value in settings.items():
        setattr(ports, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(ports, name, value)


def replay_changed(**settings):
    with changed(**settings):
        return replay(STORE)


# %%
# Raising a different type: ``differs``
# -------------------------------------
#
# The range check now raises ``TypeError``. The exception's type is part of the
# behaviour - a caller catching ``ValueError`` would no longer catch it - so the case
# differs.

report = replay_changed(RANGE_ERROR=TypeError)
show_cases(report)
print()
print(report.failures[0].describe())

# %%
# No longer raising: ``differs``
# ------------------------------
#
# The other direction - a guard that has gone missing - is the one worth having. The
# record says the call raised; now it returns ``99999``.

report = replay_changed(STRICT=False)
show_cases(report)
print()
print(report.failures[0].describe())

# %%
# Raising where it used to return: ``raised``
# -------------------------------------------
#
# The range is narrowed, so ``"8080"`` is now rejected. The record says the call
# returned, and on replay it raised - a status of its own, because it often means
# something other than a change in the answer: the call could not run outside its
# original context (a clock, a database, an environment variable), or its signature
# changed so the recorded arguments no longer fit. ``case.error`` names the exception.

report = replay_changed(MAX_PORT=1023)
show_cases(report)
print()
print(report.failures[0].describe())

# %%
# Nothing was recorded by those replays, and with every setting restored the store
# replays clean.

print(replay(STORE).summary())

# %%
# Turning exception recording off
# -------------------------------
#
# ``record_exceptions=False`` makes a raising call a non-event: not recorded, just
# propagated. Reach for it when a function raises for reasons that are not about its
# behaviour - a flaky network call, a missing local fixture. (This function is never
# replayed, so it can live in the script.)


@record_func(STORE, record_exceptions=False)
def flaky(x):
    raise ConnectionError("nothing to see here")


try:
    flaky(1)
except ConnectionError as e:
    print(f"propagated: {e}")
print("records for flaky:", len(list((STORE / "__main__").rglob("*.json"))))

# %%
# ``KeyboardInterrupt`` and ``SystemExit`` are never recorded whatever this setting says.
# They mean the process is going away, not that the function misbehaved.

# %%
# Code drift
# ----------
#
# ``code_hash`` - a hash of the function's source - is stored on every record. Replay
# runs *today's* code on recorded inputs, so the source having changed since recording
# is expected, not a failure. Replay compares the hashes and reports the answer as
# ``case.code_drift``, alongside the recorded and current hashes.
#
# Changing the source of ``ports.py`` from inside this script would leave the file
# edited for the next run, so the next cell **simulates a code change** instead: it
# rewrites the ``code_hash`` stored in one record, as if that record had been made from
# an older version of ``parse_port``.

drifted = next(p for p in record_paths(STORE) if "exception" not in p.read_text())
record = json.loads(drifted.read_text())
record["identity"]["code_hash"] = "0" * 16  # simulated: "recorded from older source"
drifted.write_text(json.dumps(record, indent=2))

report = replay(STORE)
show_cases(report)
print()
for case in report.cases:
    print(
        f"{case.status.value:<7} code_drift={case.code_drift!s:<5} "
        f"recorded={case.recorded_code_hash} current={case.current_code_hash}"
    )

# %%
# Both cases still pass: the report says the drifted one *passed across a change*,
# which is the good outcome of a refactor - the implementation changed and the behaviour
# did not. A pass without drift only says that nothing changed.
#
# ``describe()`` notes the drift too, so it shows up when a drifted case fails:

print(next(c for c in report.cases if c.code_drift).describe())

# %%
# Drift is not a failure in ``replay()`` and has no effect on ``report.passed``. Under
# ``replay_tests`` (example 14) a passing case with drift also emits a
# ``CodeDriftWarning``; to make drift fail a suite, promote the warning, in
# ``pyproject.toml``:
#
# .. code-block:: toml
#
#    [tool.pytest.ini_options]
#    filterwarnings = ["error::regression_records.CodeDriftWarning"]
#
#
# or check the column yourself after a plain replay:
#
# .. code-block:: python
#
#    drifted = [case.name for case in report.cases if case.code_drift]
