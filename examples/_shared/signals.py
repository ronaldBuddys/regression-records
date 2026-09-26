"""The functions example 9 records and replays, on stdlib ``array.array`` values.

``array.array`` is a C type with no attributes to record, so each function records only
through the leaf serializer example 9 writes. The functions record into a store with
its own copy of the default registry, ``REGISTRY``, on which example 9 registers that
serializer. In your own code, register it in the module itself, so any process that
imports the module can load the records.

``GAIN`` multiplies the samples in ``amplify``; example 9 changes it and restores it.
"""

from __future__ import annotations

from array import array

from regression_records import RecordStore, record_func
from regression_records.serialization import default_registry

from .demo import EXAMPLE_STORES

REGISTRY = default_registry.copy()
STORE = RecordStore(
    EXAMPLE_STORES / "09_custom_serializer", serializer_registry=REGISTRY
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
