"""Run every example end to end.

The examples are documentation that executes, which is the only kind that cannot quietly
go stale: an API change that breaks one fails here rather than in a reader's terminal.
Each is run in a subprocess so that one example's module-level state - decorated
functions, registered serializers, a mutated store - cannot leak into the next.

Each also runs as a copy, alone in an empty directory: the gallery offers every page as
a download, a ``.py`` or a notebook, and a page that leaned on a file beside it would
break in the reader's hands while passing here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
# Pages live one level down, in the gallery's numbered sections.
EXAMPLES = sorted(EXAMPLES_DIR.glob("*/[0-9]*.py"), key=lambda p: p.name)
SECTIONS = sorted(
    p for p in EXAMPLES_DIR.iterdir() if p.is_dir() and p.name[0].isdigit()
)

# The environment variable the example code reads its store from when imported.
STORE_ENV = "REGRESSION_RECORDS_STORE"

# A backtick-quoted span that is neither ``double`` nor an rst `link <url>`_ / :role:`x`.
MARKDOWN_INLINE_CODE = re.compile(r"(?<![`:\w])`[^`<\n]+`(?![`_])")


def test_examples_are_discovered():
    """A glob that silently found nothing would make every test below vacuous."""
    assert len(EXAMPLES) >= 15
    assert [p.name[:2] for p in EXAMPLES] == [f"{n:02d}" for n in range(1, 16)]


def test_every_section_has_a_header():
    for section in SECTIONS:
        assert (section / "GALLERY_HEADER.rst").is_file(), section.name


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(example, tmp_path):
    copy = tmp_path / example.name
    shutil.copy(example, copy)
    # Without the switch the suite's fixture sets: an example must turn recording on
    # itself, as it would for a reader running it. Without PYTHONPATH, too, so that
    # nothing but the installed package is importable. And without a store a developer
    # may have exported: a page must set its own, or importing its code fails.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"REGRESSION_RECORDS", "PYTHONPATH", STORE_ENV}
    }
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, copy.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, (
        f"{example.name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_example_sets_its_store_before_importing_code_that_records(example):
    """The code under test reads the store from the environment when imported."""
    text = example.read_text()
    assert f'os.environ["{STORE_ENV}"] = ' in text, "the page sets the store"
    assert text.index(f'os.environ["{STORE_ENV}"] = ') < text.index(
        "from regression_records._examples"
    ), "before importing the code under test"


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

    text = example.read_text()
    assert "_shared" not in text, "a page imports nothing from beside it"
    assert "plt.subplots(" in text, "every page draws a figure, its gallery thumbnail"
    if example != EXAMPLES[-1]:
        assert "# **Next:** :ref:`sphx_glr_auto_examples_" in text, (
            "every page but the last ends by linking the next"
        )

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
