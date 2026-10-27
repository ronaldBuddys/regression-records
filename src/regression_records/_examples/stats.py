"""Summary statistics, for the gallery page ``01_first_record``."""

from __future__ import annotations

import os

import numpy as np

from regression_records import record_func

# Where records go is configuration, set by whoever runs this code; never set here.
STORE = os.environ["REGRESSION_RECORDS_STORE"]


@record_func(STORE)
def summarize(values: list[float], *, scale: float = 1.0) -> dict:
    """Return the size, mean and scaled values of ``values``."""
    scaled = np.asarray(values) * scale
    return {"n": int(scaled.size), "mean": float(scaled.mean()), "values": scaled}
