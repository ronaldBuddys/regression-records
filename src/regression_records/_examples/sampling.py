"""A Monte Carlo estimate and an exact total, for the gallery page ``03_tolerance``."""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/sampling, or a fresh temporary directory.
STORE = example_store("sampling")

DRIFT = 0.0  # relative perturbation applied to every float result


@record_func(STORE)
def simulate(seed: int) -> dict:
    """Return four standard normal draws from ``seed``, and a Monte Carlo estimate."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=1000)
    return {
        "draws": draws[:4] * (1 + DRIFT),
        "estimate": float(np.abs(draws).mean()) * (1 + DRIFT),
        "seed": seed,
    }


@record_func(STORE)
def tally(amounts: list[float]) -> dict:
    """Return the total of ``amounts`` in currency, which should be exact."""
    return {"total": float(np.sum(amounts)) * (1 + DRIFT), "count": len(amounts)}
