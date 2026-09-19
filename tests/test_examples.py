"""Run every example end to end.

The examples are documentation that executes, which is the only kind that cannot quietly
go stale: an API change that breaks one fails here rather than in a reader's terminal.
Each is run in a subprocess so that one example's module-level state - decorated
functions, registered serializers, a mutated store - cannot leak into the next.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(p for p in EXAMPLES_DIR.glob("*.py"))

# A backtick-quoted span that is neither ``double`` nor an rst `link <url>`_ / :role:`x`.
MARKDOWN_INLINE_CODE = re.compile(r"(?<![`:\w])`[^`<\n]+`(?![`_])")


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
def test_example_is_written_in_the_sphinx_gallery_style(example):
    """The docs build renders these, and with -W a malformed one fails it.

    Checked here as well because this runs in seconds and says which line is wrong.
    """
    lines = example.read_text().splitlines()
    assert lines[0] == '"""', "an example starts with a module docstring"
    assert set(lines[2]) == {"="}, "with an underlined rst title"
    assert len(lines[2]) == len(lines[1]), "whose underline matches its length"
    assert any(line.startswith("# %%") for line in lines), "and is split into cells"

    for number, line in enumerate(lines, start=1):
        where = f"{example.name}:{number}"
        assert "[markdown]" not in line, (
            f"{where} is a jupytext cell, not a gallery one"
        )
        if not line.startswith("# "):
            continue
        # Prose is reStructuredText, where Markdown's forms mean something else:
        # `x` is a (usually dangling) reference, and "## Title" is a literal.
        assert not line.startswith(("# ## ", "# ### ", "# ```")), (
            f"{where} uses a Markdown heading or fence"
        )
        assert not MARKDOWN_INLINE_CODE.search(line), (
            f"{where} uses `single backticks`; rst inline code is ``double``"
        )
