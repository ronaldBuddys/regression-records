"""The functions example 4 records while it switches recording on and off.

``floor_area`` calls ``area`` once per room, so recording one records both. ``WALL``
is taken off each side of a room; example 4 changes it during a replay, then restores
it.
"""

from __future__ import annotations

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "04_switching"

WALL = 0.0  # thickness taken off each side of a room


@record_func(STORE)
def area(width: float, height: float) -> float:
    """Return the area of a ``width`` by ``height`` rectangle."""
    return width * height


@record_func(STORE)
def floor_area(rooms: list[tuple[float, float]]) -> float:
    """Return the total area of ``rooms``, each given as ``(width, height)``."""
    return sum(area(width - WALL, height - WALL) for width, height in rooms)


@record_func(STORE, enabled=False)
def perimeter(width: float, height: float) -> float:
    """Return the perimeter of a rectangle; never recorded, whatever the switch says."""
    return 2 * (width + height)
