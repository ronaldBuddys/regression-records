"""The functions example 14 records and replays under pytest.

``score`` and ``grade`` are recorded; ``weights`` is the helper beneath ``score`` that
example 14 breaks.
"""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "14_replay_tests"


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
