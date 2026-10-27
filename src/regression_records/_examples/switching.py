"""Room areas, for the gallery page ``04_modes_and_new_cases``."""

from __future__ import annotations

from regression_records import record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/switching, or a fresh temporary directory.
STORE = example_store("switching")

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
