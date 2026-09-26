"""Sample classes used across the examples.

Between them they cover the shapes a class can take, and every one is recorded by its
attributes with no configuration:

* :class:`Point` - plain attributes in ``__dict__``.
* :class:`Reading` / :class:`Experiment` - nesting, with numpy arrays inside.
* :class:`Quote` - a named tuple, recorded with its class and its fields by name.
* :class:`Vec3` - ``__slots__``, so no ``__dict__``: its slots are recorded instead.
* :class:`Calibration` - a frozen, slotted dataclass holding a read-only mapping.

:class:`LiveReading` is the exception: it holds an open file, which is not data, so it
cannot be recorded at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np


class Quote(NamedTuple):
    """A named tuple. Recorded with its class and its fields by name.

    Like every recorded class it has to be importable by name, which is why it lives
    here rather than in the example that uses it.
    """

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
        self.handle = open(os.devnull)  # noqa: SIM115 - deliberately a live resource


class Experiment:
    """Nests a ``Point`` and ``Reading`` objects - recorded recursively."""

    def __init__(self, name: str, origin: Point, readings: list[Reading]):
        self.name = name
        self.origin = origin
        self.readings = readings

    def __repr__(self):
        return f"Experiment(name={self.name!r}, origin={self.origin!r}, readings={self.readings!r})"


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
    """Frozen and slotted, with a read-only mapping of per-sensor offsets.

    ``__post_init__`` wraps ``offsets`` in a ``MappingProxyType``; loading a record
    does not run it, but the mapping is recorded as a read-only view and comes back as
    one.
    """

    instrument: str
    offsets: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "offsets", MappingProxyType(dict(self.offsets)))
