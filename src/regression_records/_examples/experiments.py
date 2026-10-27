"""A function on custom classes, for the gallery page ``09_custom_classes``."""

from __future__ import annotations

from regression_records import record_func

from . import example_store
from .models import Experiment, Point

# $REGRESSION_RECORDS_EXAMPLE_STORE/experiments, or a fresh temporary directory.
STORE = example_store("experiments")


@record_func(STORE)
def recentre(experiment: Experiment, origin: Point) -> Experiment:
    """Return a copy of ``experiment`` moved to ``origin``."""
    return Experiment(name=experiment.name, origin=origin, readings=experiment.readings)
