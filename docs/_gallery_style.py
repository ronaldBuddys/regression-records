"""Matplotlib house style for the example gallery.

A module of its own, not a function in ``conf.py``, so that ``sphinx_gallery_conf`` can
name it by string: a function object in the config cannot be pickled with the Sphinx
environment, and under ``-W`` the resulting warning fails the build.
"""

from __future__ import annotations


def house_style(gallery_conf: dict, fname: str) -> None:
    """Reset matplotlib to one style before each example, so every figure matches."""
    import matplotlib as mpl

    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.0),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
