"""Run every example end to end.

The examples are documentation that executes, which is the only kind that cannot quietly
go stale: an API change that breaks one fails here rather than in a reader's terminal.
Each is run in a subprocess so that one example's module-level state - decorated
functions, registered serializers, a mutated store - cannot leak into the next.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(p for p in EXAMPLES_DIR.glob("*.py"))


def test_examples_are_discovered():
    """A glob that silently found nothing would make every test below vacuous."""
    assert len(EXAMPLES) >= 10


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(example):
    result = subprocess.run(
        [sys.executable, str(example)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{example.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example_is_percent_format(example):
    """Each example must stay a jupytext notebook, not drift into a plain script."""
    text = example.read_text()
    assert text.startswith("# ---\n# jupyter:\n"), "missing the jupytext header"
    assert "# %% [markdown]\n" in text, "no markdown cells"
