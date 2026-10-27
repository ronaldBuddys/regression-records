"""Summary statistics, for the gallery page ``01_first_record``."""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/stats, or a fresh temporary directory.
STORE = example_store("stats")


@record_func(STORE)
def summarize(values: list[float], *, scale: float = 1.0) -> dict:
    """Return the size, mean and scaled values of ``values``."""
    scaled = np.asarray(values) * scale
    return {"n": int(scaled.size), "mean": float(scaled.mean()), "values": scaled}
