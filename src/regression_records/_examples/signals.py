"""Filters on ``array.array``, for the gallery page ``10_custom_serializer``."""

from __future__ import annotations

from array import array

from regression_records import RecordStore, record_func
from regression_records.serialization import default_registry

from . import example_store

REGISTRY = default_registry.copy()
# $REGRESSION_RECORDS_EXAMPLE_STORE/signals, or a fresh temporary directory.
STORE = RecordStore(example_store("signals"), serializer_registry=REGISTRY)

GAIN = 1.0  # applied to every sample in amplify


@record_func(STORE)
def amplify(samples: array, k: float) -> array:
    """Return ``samples`` scaled by ``k`` (and by ``GAIN``)."""
    return array(samples.typecode, (x * k * GAIN for x in samples))


@record_func(STORE)
def decimate(samples: array, factor: int) -> array:
    """Return every ``factor``-th sample."""
    return samples[::factor]
