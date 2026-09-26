"""The pipeline example 15 records, under run manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from regression_records import RecordStore, record_func, recording, session

from .demo import EXAMPLE_STORES

STORE = RecordStore(EXAMPLE_STORES / "15_run_manifests")


@record_func(STORE)
def daily_pnl(prices: np.ndarray, position: float) -> np.ndarray:
    """Return the daily PnL of holding ``position`` units through ``prices``."""
    return position * np.diff(prices)


def run_pipeline(prices: list[float], data_version: int) -> Path:
    """Record one run of a toy pipeline, and return its manifest's path.

    ``data_version`` stands in for the version of the input data a real pipeline would
    read: the environment cannot know it, so it goes in the session's context.
    """
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
