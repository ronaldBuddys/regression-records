"""A discount curve, for the gallery page ``02_catching_a_regression``."""

from __future__ import annotations

import numpy as np

from regression_records import record_func


def zero_rate(tenor: np.ndarray) -> np.ndarray:
    """Return the zero rate for each tenor, in years: 2% short, 3.5% long."""
    return 0.02 + 0.015 * (1 - np.exp(-tenor / 4))


@record_func()
def discount_curve(tenors: list[float]) -> dict:
    """Return the tenors, their zero rates and their discount factors."""
    t = np.asarray(tenors, dtype=float)
    zero = zero_rate(t)
    return {"tenors": t, "zero": zero, "discount": np.exp(-zero * t)}
