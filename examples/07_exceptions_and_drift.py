"""
Exceptions, and code drift
==========================

Raising is behaviour too. A function that used to reject bad input and now silently
accepts it has regressed just as surely as one that returns the wrong number, so
exceptions are recorded and compared alongside return values.
"""

import sys
import warnings
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared.demo import fresh_store, show_record, show_result

from regression_records import CodeDriftWarning, RegressionMismatchError, record_func

STORE = fresh_store("07_exceptions_and_drift")

STRICT = True


@record_func(STORE)
def parse_port(text):
    port = int(text)
    if STRICT and not 0 < port < 65536:
        raise ValueError(f"port out of range: {port}")
    return port


print("ok:", parse_port("8080"))

try:
    parse_port("99999")
except ValueError as e:
    print("raised and recorded:", e)

# %%
# What a recorded exception holds
# -------------------------------
#
# Exactly one of ``output`` / ``exception`` is set on a record. The exception's type, message
# and formatted traceback are all stored - but only the **type** takes part in the
# comparison. Messages routinely carry paths, addresses and timings that would turn every
# run into a false mismatch; the message and traceback are there for you to read.

show_record(STORE, keys=["exception"], select=lambda r: "exception" in r)

# %%
# Validating against it
# ---------------------
#
# Raising the same type again passes. The exception still propagates - recording never
# swallows or substitutes what a function does.

try:
    parse_port("99999")
except ValueError:
    print("raised again, comparison passed:", parse_port.last_result.passed)

# %%
# Raising a different type
# ------------------------


@record_func(STORE, name="parse_port")
def parse_port_typeerror(text):
    port = int(text)
    if not 0 < port < 65536:
        raise TypeError(f"port out of range: {port}")
    return port


with warnings.catch_warnings():
    warnings.simplefilter("ignore", CodeDriftWarning)
    try:
        parse_port_typeerror("99999")
    except TypeError:
        show_result(parse_port_typeerror.last_result)

# %%
# **The function's own exception reaches you, not the mismatch.** When a call both raises
# and fails its comparison, the real exception is the more direct signal and is what
# propagates; the mismatch waits on ``last_result``. A test suite still fails either way,
# because the exception is not caught.
#
# No longer raising
# -----------------
#
# The other direction - a guard that has gone missing - is the one worth having.

STRICT = False

with warnings.catch_warnings():
    warnings.simplefilter("ignore", CodeDriftWarning)
    try:
        parse_port("99999")
    except RegressionMismatchError as e:
        show_result(e.result)

STRICT = True

# %%
# Here nothing was raised by the function, so there is no "real exception" to prefer and
# the mismatch itself is raised.
#
# Turning exception recording off
# -------------------------------
#
# ``record_exceptions=False`` makes a raising call a non-event: not recorded, not validated,
# just propagated. Reach for it when a function raises for reasons that are not about its
# behaviour - a flaky network call, a missing local fixture.


@record_func(STORE, record_exceptions=False)
def flaky(x):
    raise ConnectionError("nothing to see here")


try:
    flaky(1)
except ConnectionError as e:
    print(f"propagated: {e}")
print("records for `flaky`:", len(list((STORE / "__main__" / "flaky").rglob("*.json"))))

# %%
# ``KeyboardInterrupt`` and ``SystemExit`` are never recorded whatever this setting says. They
# mean the process is going away, not that the function misbehaved.

# %%
# Code drift
# ----------
#
# ``code_hash`` - a hash of the function's source - is stored on every record but is
# deliberately **not** part of the lookup key, so editing a function never orphans its
# history. Instead, validating against a record captured from different source emits a
# ``CodeDriftWarning``.
#
# It is not a failure. It answers a question a passing comparison otherwise leaves open:
# does this pass mean *"behaviour held across a change"* or merely *"nothing changed"*?


@record_func(STORE, name="parse_port")
def parse_port_refactored(text):
    port = int(text)  # same behaviour, different source
    limit = 65536
    if STRICT and not 0 < port < limit:
        raise ValueError(f"port out of range: {port}")
    return port


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    parse_port_refactored("8080")

print("passed:", parse_port_refactored.last_result.passed)
for w in caught:
    print(f"{w.category.__name__}: {w.message}")

# %%
# A passing comparison *with* a drift warning is the good outcome of a refactor: the
# implementation changed and the behaviour did not.
#
# To make drift a hard failure in a suite, promote the warning:
#
# .. code-block:: python
#
#    import warnings
#    from regression_records import CodeDriftWarning
#
#    warnings.simplefilter("error", CodeDriftWarning)
#
#
# or, in ``pyproject.toml``:
#
# .. code-block:: toml
#
#    [tool.pytest.ini_options]
#    filterwarnings = ["error::regression_records.CodeDriftWarning"]
#
