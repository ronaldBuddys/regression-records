"""Filters on ``array.array``, for the gallery page ``10_custom_serializer``."""

from __future__ import annotations

import os
from array import array

from regression_records import RecordStore, record_func
from regression_records.serialization import default_registry

REGISTRY = default_registry.copy()
# Where records go is configuration, set by whoever runs this code; never set here.
STORE = RecordStore(
    os.environ["REGRESSION_RECORDS_STORE"], serializer_registry=REGISTRY
)

GAIN = 1.0  # applied to every sample in amplify


@record_func(STORE)
def amplify(samples: array, k: float) -> array:
    """Return ``samples`` scaled by ``k`` (and by ``GAIN``)."""
    return array(samples.typecode, (x * k * GAIN for x in samples))


@record_func(STORE)
def decimate(samples: array, factor: int) -> array:
    """Return every ``factor``-th sample."""
    return samples[::factor]
