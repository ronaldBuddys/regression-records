"""Weighted scores and letter grades, for the gallery page ``14_replay_tests``."""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/scoring, or a fresh temporary directory.
STORE = example_store("scoring")


def weights(n: int) -> np.ndarray:
    """Return ``n`` linearly increasing weights that sum to one."""
    raw = np.arange(1, n + 1, dtype=float)
    return raw / raw.sum()


@record_func(STORE)
def score(values: list[float]) -> float:
    """Return the weighted average of ``values``, later values weighted more."""
    return float(np.dot(values, weights(len(values))))


@record_func(STORE)
def grade(value: float) -> str:
    """Return a letter grade for a score between 0 and 1."""
    return "A" if value >= 0.8 else "B" if value >= 0.5 else "C"
