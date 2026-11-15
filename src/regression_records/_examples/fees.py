"""Trading commission, for the gallery page ``05_accepting_a_change``."""

from __future__ import annotations

import numpy as np

from regression_records import record_func

RATE = 0.10  # commission, as a fraction of the traded amount


@record_func()
def commission(amounts: list[float]) -> np.ndarray:
    """Return the commission on each of ``amounts``, rounded to the cent."""
    return np.round(np.asarray(amounts) * RATE, 2)
