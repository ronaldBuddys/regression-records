"""The function example 5 records, changes on purpose, and re-records.

``RATE`` is the commission rate. Example 5 raises it to stand in for an intended edit
to this file, and restores it afterwards.
"""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "05_accepting_a_change"

RATE = 0.10  # commission, as a fraction of the traded amount


@record_func(STORE)
def commission(amounts: list[float]) -> np.ndarray:
    """Return the commission on each of ``amounts``, rounded to the cent."""
    return np.round(np.asarray(amounts) * RATE, 2)
