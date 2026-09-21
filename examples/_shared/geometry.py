"""The function example 9 records and replays: scaling a ``Vec3``.

``Vec3`` has no dict-shaped state, so it records only through a custom serializer.
``scale`` records into a store with its own copy of the default registry, ``REGISTRY``,
on which example 9 registers the serializer it writes. In your own code, register it
in the module itself, so any process that imports the module can load the records.

``FLIP`` multiplies the ``z`` component; example 9 changes it and restores it.
"""

from __future__ import annotations

from regression_records import RecordStore, record_func
from regression_records.serialization import default_registry

from .demo import EXAMPLE_STORES
from .models import Vec3

REGISTRY = default_registry.copy()
STORE = RecordStore(
    EXAMPLE_STORES / "09_custom_serializer", serializer_registry=REGISTRY
)

FLIP = -1.0  # applied to the z component


@record_func(STORE)
def scale(v: Vec3, k: float) -> Vec3:
    """Return ``v`` scaled by ``k``, reflected in the xy-plane."""
    return Vec3(v.x * k, v.y * k, v.z * k * FLIP)
