"""The function example 1 records: summary statistics of a sample.

Replay imports a recorded function by its module and qualified name, so it has to live
in a module like this one. Defined in the example script itself it would belong to
``__main__``, which replay cannot import again.
"""

from __future__ import annotations

import numpy as np

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "01_first_record"


@record_func(STORE)
def summarize(values: list[float], *, scale: float = 1.0) -> dict:
    """Return the size, mean and scaled values of ``values``."""
    scaled = np.asarray(values) * scale
    return {"n": int(scaled.size), "mean": float(scaled.mean()), "values": scaled}
