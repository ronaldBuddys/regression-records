"""The function example 8 records: custom objects as arguments and return value."""

from __future__ import annotations

from regression_records import record_func

from .demo import EXAMPLE_STORES
from .models import Experiment, Point

STORE = EXAMPLE_STORES / "08_custom_classes"


@record_func(STORE)
def recentre(experiment: Experiment, origin: Point) -> Experiment:
    """Return a copy of ``experiment`` moved to ``origin``."""
    return Experiment(name=experiment.name, origin=origin, readings=experiment.readings)
