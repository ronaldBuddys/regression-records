"""
replay_tests : one pytest test per record
=========================================

``replay()`` returns one report for a whole store. In a test suite you usually want
something finer: one test per record, so that a failure names the case that moved and
the rest of the store still reports as passing.

``replay_tests`` builds exactly that - a pytest test function, parametrized over the
records in a store. This example records a few calls, writes a small test module around
``replay_tests``, runs pytest on it, then breaks a helper and runs it again.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import _shared
from _shared import scoring
from _shared.demo import fresh_store

from regression_records import enable

STORE = fresh_store(scoring.STORE.name)

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# Record some calls
# -----------------
#
# ``score`` and ``grade`` are recorded; ``weights`` is the helper beneath ``score``.
# They live in ``examples/_shared/scoring.py`` so that replay - here, inside pytest - can
# import them.

print(inspect.getsource(scoring))

for values in ([0.9, 0.8, 1.0], [0.1, 0.5]):
    print(values, "->", scoring.grade(scoring.score(values)))

# %%
# The test module
# ---------------
#
# This is the whole test file:

SUITE = fresh_store("14_replay_tests_suite")
TEST_FILE = SUITE / "test_regressions.py"
TEST_FILE.write_text(
    "from regression_records import replay_tests\n"
    "\n"
    f"test_regressions = replay_tests({str(STORE)!r})\n"
)
print(TEST_FILE.read_text())

# %%
# **The name must start with** ``test_``. pytest collects functions by name, and
# ``replay_tests`` returns an ordinary function; bound to any other name it is silently
# never run.
#
# In a real project the store path is relative to the repository, as in
# ``replay_tests("tests/regression_data")``, and the file sits beside your other tests.
# Keyword arguments are those of ``replay()``: ``tolerance``, ``tolerance_overrides``,
# ``select``, ``comparator_registry`` and ``max_diffs``.
#
# Running it
# ----------
#
# pytest runs in a subprocess here, as it would from a terminal or CI.
# ``PYTHONPATH`` points at the examples directory, so the test process can import
# ``_shared.scoring`` as replay requires; in your project the package under test is
# installed and importable already.

EXAMPLES_DIR = Path(_shared.__file__).resolve().parent.parent


def run_pytest():
    """Run pytest on the suite directory and print what it prints."""
    path = [str(EXAMPLES_DIR), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, path))}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--no-header", "-p", "no:cacheprovider"],
        cwd=SUITE,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout)
    return result.returncode


print("exit code:", run_pytest())

# %%
# One test per record, each identified as ``module:qualname@<record id>``: the store's
# four records became four tests. A new record is picked up the next time pytest
# collects the module, with no change to the test file.
#
# Breaking a helper
# -----------------
#
# The test process imports ``scoring`` afresh, so a change made in this script would
# not reach it. A ``conftest.py`` that swaps ``weights`` for a broken version - equal
# weights instead of increasing ones - stands in for an edit to ``scoring.py``, and is
# deleted afterwards.

CONFTEST = SUITE / "conftest.py"
CONFTEST.write_text(
    "import numpy as np\n"
    "from _shared import scoring\n"
    "\n"
    "scoring.weights = lambda n: np.full(n, 1.0 / n)\n"
)
try:
    print("exit code:", run_pytest())
finally:
    CONFTEST.unlink()

# %%
# The two ``score`` cases fail, each with its own message: the case's ``describe()``,
# naming the function, the record and the value that moved. The ``grade`` cases still
# pass - they replay the recorded scores, so a change upstream of their arguments does
# not reach them.
#
# What else to know
# -----------------
#
# * A record that cannot be loaded, imported or called **fails** its test; it is never
#   skipped, so a store cannot quietly stop being checked.
# * A case that passes although the function's source has changed since recording
#   (example 7) passes with a ``CodeDriftWarning``, which a ``filterwarnings`` entry can
#   turn into a failure.
# * The store is read when pytest collects the module, and a store path that does not
#   exist is a collection error, not an empty pass.
