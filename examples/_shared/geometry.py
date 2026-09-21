"""The functions example 9 records and replays: on ``Vec3`` and ``Waveform``.

Neither class has dict-shaped state, so each records only through a custom serializer.
The functions record into a store with its own copy of the default registry,
``REGISTRY``, on which example 9 registers the serializers it writes. In your own code,
register them in the module itself, so any process that imports the module can load
the records.

``FLIP`` multiplies the ``z`` component in ``scale``; example 9 changes it and restores
it.
"""

from __future__ import annotations

from regression_records import RecordStore, record_func
from regression_records.serialization import default_registry

from .demo import EXAMPLE_STORES
from .models import Vec3, Waveform

REGISTRY = default_registry.copy()
STORE = RecordStore(
    EXAMPLE_STORES / "09_custom_serializer", serializer_registry=REGISTRY
)

FLIP = -1.0  # applied to the z component


@record_func(STORE)
def scale(v: Vec3, k: float) -> Vec3:
    """Return ``v`` scaled by ``k``, reflected in the xy-plane."""
    return Vec3(v.x * k, v.y * k, v.z * k * FLIP)


@record_func(STORE)
def reflect(v: Vec3) -> Vec3:
    """Return ``v`` reflected through the origin."""
    return Vec3(-v.x, -v.y, -v.z)


@record_func(STORE)
def decimate(wave: Waveform, factor: int) -> Waveform:
    """Return every ``factor``-th sample of ``wave``, at the correspondingly lower rate."""
    return Waveform(wave.samples[::factor], wave.rate // factor)
