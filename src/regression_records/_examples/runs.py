"""Daily PnL, recorded under a session, for the gallery page ``06_run_manifests``."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from regression_records import RecordStore, record_func, recording, session

# Where records go is configuration, set by whoever runs this code; never set here.
STORE = RecordStore(os.environ["REGRESSION_RECORDS_STORE"])


@record_func(STORE)
def daily_pnl(prices: np.ndarray, position: float) -> np.ndarray:
    """Return the daily PnL of holding ``position`` units through ``prices``."""
    return position * np.diff(prices)


def run_pipeline(prices: list[float], data_version: int) -> Path:
    """Record one run of a toy pipeline, and return its manifest's path."""
    with (
        recording(),
        session(
            STORE,
            context={"data_version": data_version, "seed": 0},
            packages=["numpy"],
        ) as run,
    ):
        for position in (10.0, -5.0):
            daily_pnl(np.asarray(prices), position)
    assert run.manifest_path is not None
    return run.manifest_path
