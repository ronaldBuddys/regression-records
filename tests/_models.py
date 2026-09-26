"""Stand-ins for the workspace classes that must be stored by their attributes.

regression-records cannot depend on labelled-array or mktdata-structs, so these mirror
the shapes that made those classes unstorable before 0.4.0, not the classes themselves:

* :class:`Labels` - a ``__slots__`` ``Mapping`` with lazily filled lookup caches keyed
  by numpy scalars, like ``labelled_array.Dimensions``;
* :class:`Grid` / :class:`Surface` - a ``__dict__`` class chain holding a dict of arrays,
  a :class:`Labels` and a backend object, like ``LabelledArray`` and ``MktVolSurface``;
* :class:`Universe` - a frozen slotted dataclass holding a ``mappingproxy`` and a
  read-only array, like ``mktdata_structs.MarketDataUniverse``.

They live in an importable module because a recorded class is re-imported by name.
"""

from __future__ import annotations

import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np


def _frozen(values) -> np.ndarray:
    array = np.array(values)
    array.setflags(write=False)
    return array


class Labels(Mapping[str, np.ndarray]):
    """Named label arrays, with a per-name position cache filled on first lookup."""

    __slots__ = ("_labels", "_names", "_lookups")

    def __init__(self, **labels) -> None:
        self._labels = {name: _frozen(values) for name, values in labels.items()}
        self._names = tuple(labels)
        self._lookups: dict[str, dict] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        return self._labels[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def position(self, name: str, label) -> int:
        # Keys are the array's own scalars (np.datetime64, np.int64, ...), not strings.
        if name not in self._lookups:
            self._lookups[name] = {v: i for i, v in enumerate(self._labels[name])}
        return self._lookups[name][label]

    def __eq__(self, other) -> bool:
        # Deliberately loose, to show comparison walks attributes instead of this.
        return isinstance(other, Labels) and self._names == other._names

    __hash__ = None


class Backend:
    """An array backend with no state of its own, like labelled-array's plugins."""

    name: ClassVar[str] = "numpy"


class Grid:
    """Data variables over labelled dimensions, stored in an instance ``__dict__``."""

    def __init__(self, data: dict, labels: Labels, attrs: dict | None = None) -> None:
        self._backend = Backend()
        self._data = {name: np.asarray(values) for name, values in data.items()}
        self._labels = labels
        self._attrs = dict(attrs or {})


class Surface(Grid):
    """A subclass that adds only class-level configuration, like ``MktVolSurface``."""

    value_name: ClassVar[str] = "implied_vol"


@dataclass(frozen=True, slots=True)
class Universe:
    """A frozen, slotted bundle of market objects."""

    symbol: str
    surface: Surface
    curves: Mapping[str, Grid]
    snap_days: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "curves", types.MappingProxyType(dict(self.curves)))
        object.__setattr__(self, "snap_days", _frozen(self.snap_days))
        if not len(self.snap_days):
            raise ValueError("a universe needs at least one snap")


class Vec:
    """A plain ``__slots__`` class, with a name-mangled slot and one left unset."""

    __slots__ = ("x", "__secret", "maybe")

    def __init__(self, x: float, secret: str) -> None:
        self.x = x
        self.__secret = secret

    @property
    def secret(self) -> str:
        return self.__secret


class Quote(NamedTuple):
    """A named tuple with a derived property, which is not stored."""

    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class Holder:
    """Holds whatever a test gives it: a function, a lock, an open file."""

    held: object


def spx_like_universe() -> Universe:
    """Build a small universe with every shape above, caches already filled."""
    trade_dates = np.array(["2026-01-02", "2026-01-05"], dtype="datetime64[D]")
    labels = Labels(trade_date=trade_dates, strike=np.array([90, 100, 110]))
    labels.position("trade_date", trade_dates[1])
    labels.position("strike", np.int64(100))
    surface = Surface(
        {"implied_vol": np.linspace(0.1, 0.3, 6).reshape(2, 3)},
        labels,
        attrs={"source": "stand-in", "snaps": trade_dates},
    )
    zero = Grid({"zero_rate": np.array([0.04, 0.041])}, Labels(tenor=[1, 2]))
    return Universe("SPX", surface, {"USD": zero}, np.array([0.0, 3.0]))
