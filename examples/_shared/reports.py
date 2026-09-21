"""The function example 2 records, and the helper it breaks.

``report`` is recorded; ``weight`` is the code underneath it. Example 2 swaps
``weight`` for a broken version, which is what a real regression usually looks like:
the recorded function is untouched, something it calls has changed.
"""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "02_catching_a_regression"


def weight(values: list[float]) -> dict:
    """Return ``values`` doubled, with their total and peak."""
    weighted = np.asarray(values) * 2.0
    return {
        "total": float(weighted.sum()),
        "peak": float(weighted.max()),
        "series": weighted,
    }


@record_func(STORE)
def report(values: list[float]) -> dict:
    """Return the weighted summary of ``values``, labelled."""
    return {**weight(values), "label": "weighted"}
