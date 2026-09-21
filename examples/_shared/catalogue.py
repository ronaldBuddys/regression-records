"""The function example 11 records, into a ``RecordStore`` it then inspects."""

from __future__ import annotations

import numpy as np

from regression_records import RecordStore, record_func

from .demo import EXAMPLE_STORES

STORE = RecordStore(EXAMPLE_STORES / "11_inspecting_the_store")


@record_func(STORE, tags={"suite": "pricing", "owner": "risk-team"})
def price(quantity: int, unit: float) -> dict[str, object]:
    """Return the total for ``quantity`` items at ``unit`` each, and the per-item line."""
    return {"total": quantity * unit, "detail": np.full(quantity, unit, dtype=float)}
