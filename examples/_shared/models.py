"""Sample classes used across the examples.

Between them they cover the shapes a custom type can take:

* :class:`Point` - plain attributes in ``__dict__``, recorded with no configuration.
* :class:`Reading` - a ``__getstate__``/``__setstate__`` pair, so the recorded state is
  what the class says it is rather than everything it happens to hold.
* :class:`Quote` - a named tuple, recorded with its class.
* :class:`Vec3` / :class:`Waveform` - ``__slots__``, so there is no ``__dict__`` to
  record and a custom ``Serializer`` is required.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Quote(NamedTuple):
    """A named tuple. Recorded with its class, so it comes back as a ``Quote``.

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
    """A measurement holding a live file handle that must not be recorded.

    ``__getstate__`` names the two attributes that describe this object; ``__setstate__``
    puts the handle back as ``None``. Recording follows the same protocol as ``pickle``
    here, so a class that already round-trips through pickle needs nothing new.
    """

    def __init__(self, sensor: str, samples: np.ndarray):
        self.sensor = sensor
        self.samples = samples
        self.handle = open("/dev/null")  # noqa: SIM115 - deliberately a live resource

    def __getstate__(self):
        return {"sensor": self.sensor, "samples": self.samples}

    def __setstate__(self, state):
        self.sensor = state["sensor"]
        self.samples = state["samples"]
        self.handle = None

    def __repr__(self):
        return f"Reading(sensor={self.sensor!r}, samples={self.samples!r}, handle={self.handle!r})"


class Experiment:
    """Nests the other two, plus a numpy array - recorded recursively."""

    def __init__(self, name: str, origin: Point, readings: list[Reading]):
        self.name = name
        self.origin = origin
        self.readings = readings

    def __repr__(self):
        return f"Experiment(name={self.name!r}, origin={self.origin!r}, readings={self.readings!r})"


class Vec3:
    """``__slots__``, so no ``__dict__``: needs a custom Serializer (``09_custom_serializer``)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return isinstance(other, Vec3) and self.as_tuple() == other.as_tuple()

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Vec3{self.as_tuple()}"


class Waveform:
    """A ``__slots__`` class whose payload is large enough to belong in a side file."""

    __slots__ = ("samples", "rate")

    def __init__(self, samples: np.ndarray, rate: int):
        self.samples = samples
        self.rate = rate

    def __eq__(self, other):
        return (
            isinstance(other, Waveform)
            and self.rate == other.rate
            and np.array_equal(self.samples, other.samples)
        )

    def __repr__(self):
        return f"Waveform(rate={self.rate}, n={self.samples.size})"
