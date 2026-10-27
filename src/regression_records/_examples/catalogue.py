"""Tagged catalogue prices, for the gallery page ``12_inspecting_the_store``."""

from __future__ import annotations

import os

import numpy as np

from regression_records import RecordStore, record_func

# Where records go is configuration, set by whoever runs this code; never set here.
STORE = RecordStore(os.environ["REGRESSION_RECORDS_STORE"])


@record_func(STORE, tags={"suite": "pricing", "owner": "risk-team"})
def price(quantity: int, unit: float, *, currency: str = "GBP") -> dict[str, object]:
    """Return the total for ``quantity`` items at ``unit`` each, and the per-item line."""
    return {
        "total": quantity * unit,
        "currency": currency,
        "detail": np.full(quantity, unit, dtype=float),
    }
