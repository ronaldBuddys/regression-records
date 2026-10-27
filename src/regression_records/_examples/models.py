"""Classes of every shape, for the gallery page ``09_custom_classes``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np


class Quote(NamedTuple):
    """A named tuple. Recorded with its class and its fields by name."""

    bid: float
    ask: float


class Point:
    """Plain attributes. Recorded straight out of ``__dict__``."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f"Point(x={self.x!r}, y={self.y!r})"


class Reading:
    """A named series of samples."""

    def __init__(self, sensor: str, samples: np.ndarray):
        self.sensor = sensor
        self.samples = samples

    def __repr__(self):
        return f"Reading(sensor={self.sensor!r}, samples={self.samples!r})"


class LiveReading(Reading):
    """A reading that also holds a live file handle, which has no place in a record."""

    def __init__(self, sensor: str, samples: np.ndarray):
        super().__init__(sensor, samples)
        self.handle = open(os.devnull)


class Experiment:
    """Nests a ``Point`` and ``Reading`` objects - recorded recursively."""

    def __init__(self, name: str, origin: Point, readings: list[Reading]):
        self.name = name
        self.origin = origin
        self.readings = readings

    def __repr__(self):
        return (
            f"Experiment(name={self.name!r}, origin={self.origin!r}, "
            f"readings={self.readings!r})"
        )


class Vec3:
    """``__slots__``, so no ``__dict__``: each slot is recorded as an attribute."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vec3({self.x!r}, {self.y!r}, {self.z!r})"


@dataclass(frozen=True, slots=True)
class Calibration:
    """Frozen and slotted, with a read-only mapping of per-sensor offsets."""

    instrument: str
    offsets: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "offsets", MappingProxyType(dict(self.offsets)))
