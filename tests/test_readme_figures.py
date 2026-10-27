"""The README's figures come from a script; check it still runs.

The script runs the example that draws the figure, so this also fails when the example
changes in a way the script no longer fits. Bytes are not compared: fonts and library
versions move pixels without anything being wrong.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "docs" / "_static" / "readme" / "make_figures.py"


def test_make_figures_writes_png_and_svg(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "REGRESSION_RECORDS"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    for name in ("caught-regression.png", "caught-regression.svg"):
        assert (tmp_path / name).stat().st_size > 10_000, name


def test_the_committed_figures_exist():
    """The README and the docs landing page reference these by relative path."""
    for name in ("caught-regression.png", "caught-regression.svg"):
        assert (SCRIPT.parent / name).is_file(), name
    assert (ROOT / "docs" / "_static" / "how-it-works.svg").is_file()
