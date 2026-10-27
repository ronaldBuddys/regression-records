"""Tagged catalogue prices, for the gallery page ``12_inspecting_the_store``."""

from __future__ import annotations

import numpy as np

from regression_records import RecordStore, record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/catalogue, or a fresh temporary directory.
STORE = RecordStore(example_store("catalogue"))


@record_func(STORE, tags={"suite": "pricing", "owner": "risk-team"})
def price(quantity: int, unit: float, *, currency: str = "GBP") -> dict[str, object]:
    """Return the total for ``quantity`` items at ``unit`` each, and the per-item line."""
    return {
        "total": quantity * unit,
        "currency": currency,
        "detail": np.full(quantity, unit, dtype=float),
    }
