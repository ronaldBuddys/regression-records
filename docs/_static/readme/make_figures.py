"""Render the README's figures from the example that produces them.

The caught-regression figure on the README and the docs landing page is the hero figure
of the "catching a regression" example, rendered by running that example itself: the
figure cannot drift from the code it illustrates, and the example's asserts check what
it shows. Rerun after changing the example, and commit the output::

    uv run python docs/_static/readme/make_figures.py

An optional argument names the output directory, which defaults to this script's own.
"""

from __future__ import annotations

import logging
import os
import runpy
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - after selecting the backend

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent
EXAMPLE = DOCS.parent / "examples" / "01_basics" / "02_catching_a_regression.py"

logger = logging.getLogger(__name__)


def main(out_dir: Path = HERE) -> list[Path]:
    """Run the example and save its first figure as PNG and SVG in ``out_dir``.

    Parameters
    ----------
    out_dir : Path, default this script's directory
        Where to write ``caught-regression.png`` and ``caught-regression.svg``.

    Returns
    -------
    list of Path
        The files written.
    """
    # The gallery's own style, so the README matches the docs.
    sys.path.insert(0, str(DOCS))
    from _gallery_style import house_style

    house_style({}, str(EXAMPLE))
    # Stable output, so a rerun with nothing changed leaves git clean.
    matplotlib.rcParams["svg.hashsalt"] = "regression-records"

    with tempfile.TemporaryDirectory() as scratch:
        # The example writes its store to a temporary directory of its own; running it
        # from an empty one as well keeps anything else it might write out of the repo.
        sys.argv = [str(EXAMPLE)]
        cwd = Path.cwd()
        try:
            os.chdir(scratch)
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        finally:
            os.chdir(cwd)

    figure = plt.figure(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, metadata in ((".png", {"Software": None}), (".svg", {"Date": None})):
        path = out_dir / f"caught-regression{suffix}"
        figure.savefig(path, dpi=150, bbox_inches="tight", metadata=metadata)
        written.append(path)
        logger.info("wrote %s", path)
    plt.close("all")
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
