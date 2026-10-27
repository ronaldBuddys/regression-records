"""A function on custom classes, for the gallery page ``09_custom_classes``."""

from __future__ import annotations

import os
from pathlib import Path

from regression_records import record_func

from .models import Experiment, Point

# Where records go is configuration, set by whoever runs this code; never set here.
STORE = Path(os.environ["REGRESSION_RECORDS_STORE"])


@record_func(STORE)
def recentre(experiment: Experiment, origin: Point) -> Experiment:
    """Return a copy of ``experiment`` moved to ``origin``."""
    return Experiment(name=experiment.name, origin=origin, readings=experiment.readings)
