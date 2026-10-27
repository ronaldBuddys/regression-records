"""
``replay_tests`` : one pytest test per record
=============================================

``replay()`` returns one report for a whole store. In a test suite you usually want
something finer: one test per record, so that a failure names the case that moved and
the rest of the store still reports as passing.

``replay_tests`` builds exactly that - a pytest test function, parametrized over the
records in a store. This page records a few calls, writes a small test module around
``replay_tests``, runs pytest on it, then breaks a helper and runs it again.
"""

# %%
# Setup: where records go
# -----------------------
#
# The code under test reads its store from the ``REGRESSION_RECORDS_STORE`` environment
# variable when it is imported, so the variable is set first
# (:ref:`sphx_glr_auto_examples_01_basics_01_first_record.py` says why).

import os
import tempfile

# In a project: REGRESSION_RECORDS_STORE=tests/regression_data in a .env file, loaded
# with python-dotenv's load_dotenv(). Here: a fresh temporary directory, empty each run.
os.environ["REGRESSION_RECORDS_STORE"] = tempfile.mkdtemp(prefix="regression-records-")

# %%
# Setup: the code under test
# --------------------------
#
# Replay imports a recorded function by module and name - here, inside pytest - so the
# code under test lives in a module, ``scoring``, imported from
# ``regression_records._examples`` - support code that ships with the package so that a
# page can import real code rather than define it in a cell
# (:ref:`sphx_glr_auto_examples_01_basics_01_first_record.py` says more). In your
# project, this is your own package. Its source, printed below:
#
# ``score`` and ``grade`` are recorded; ``weights`` is the helper beneath ``score``.

import inspect
import re
import subprocess
import sys
from pathlib import Path

from regression_records._examples import scoring

print(inspect.getsource(scoring))

# %%

WORKDIR = Path(tempfile.mkdtemp(prefix="regression-records-"))
STORE = scoring.STORE
print("store:", STORE)

# %%
# Record some calls
# -----------------
#
# Recording is off unless switched on; this page switches it on for the whole script.

from regression_records import enable

enable()

for values in ([0.9, 0.8, 1.0], [0.1, 0.5]):
    print(values, "->", scoring.grade(scoring.score(values)))
assert len(list(STORE.rglob("rec-*.json"))) == 4

# %%
# The test module
# ---------------
#
# This is the whole test file:

SUITE = WORKDIR / "suite"
SUITE.mkdir()
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
# pytest runs in a subprocess here, as it would from a terminal or CI. Being an
# installed module, ``scoring`` is importable there as replay requires, exactly as
# your own package is importable in your project's test process.

OUTCOME = re.compile(
    r"::test_regressions\[(?P<case>[^\]]+)\] (?P<outcome>PASSED|FAILED)"
)


def run_pytest() -> tuple[int, dict[str, str]]:
    """Run pytest on the suite, print its output, and return each test's outcome."""
    env = dict(os.environ)
    env.pop("REGRESSION_RECORDS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--no-header", "-p", "no:cacheprovider"],
        cwd=SUITE,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout)
    outcomes = {
        m["case"]: m["outcome"].lower() for m in OUTCOME.finditer(result.stdout)
    }
    return result.returncode, outcomes


code, before = run_pytest()
assert code == 0 and list(before.values()) == ["passed"] * 4

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
    "from regression_records._examples import scoring\n"
    "\n"
    "scoring.weights = lambda n: np.full(n, 1.0 / n)\n"
)
try:
    code, after = run_pytest()
finally:
    CONFTEST.unlink()

failed = sorted(case for case, outcome in after.items() if outcome == "failed")
assert code == 1 and len(after) == 4
assert [case.split("@")[0] for case in failed] == [
    "regression_records._examples.scoring:score",
    "regression_records._examples.scoring:score",
]

# %%
# The two ``score`` cases fail, each with its own message: the case's ``describe()``,
# naming the function, the record and the value that moved. The ``grade`` cases still
# pass - they replay the recorded scores, so a change upstream of their arguments does
# not reach them.

import matplotlib.pyplot as plt

cases = sorted(before)
colors = {"passed": "#0ca30c", "failed": "#d03b3b"}
fig, ax = plt.subplots(figsize=(7.5, 2.6))
for row, outcomes in enumerate([before, after]):
    for col, case in enumerate(cases):
        outcome = outcomes[case]
        ax.add_patch(
            plt.Rectangle(
                (col + 0.04, -row - 0.42), 0.92, 0.84, color=colors[outcome], lw=0
            )
        )
        ax.text(
            col + 0.5,
            -row,
            outcome,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
ax.set_xlim(0, len(cases))
ax.set_ylim(-1.5, 0.5)
ax.set_xticks(
    [c + 0.5 for c in range(len(cases))],
    [case.replace("@", "\n@") for case in cases],
    fontsize=8.5,
    family="monospace",
)
ax.set_yticks([0, -1], ["as recorded", "weights broken"])
ax.tick_params(length=0)
ax.grid(False)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title("One pytest test per record", loc="left")
fig.tight_layout()

# %%
# What else to know
# -----------------
#
# * A record that cannot be loaded, imported or called **fails** its test; it is never
#   skipped, so a store cannot quietly stop being checked.
# * A case that passes although the function's source has changed since recording
#   (:ref:`sphx_glr_auto_examples_03_what_gets_recorded_08_exceptions_and_drift.py`)
#   passes with a ``CodeDriftWarning``, which a ``filterwarnings`` entry can turn into a
#   failure.
# * The store is read when pytest collects the module, and a store path that does not
#   exist is a collection error, not an empty pass.

# %%
# **Next:** :ref:`sphx_glr_auto_examples_05_gotchas_15_gotchas.py` - what a replay
# cannot see.
