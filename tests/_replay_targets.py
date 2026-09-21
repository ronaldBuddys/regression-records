"""Importable recorded functions for the replay tests.

Replay imports a function by its recorded module and qualname, so these must live at
module level of an importable module. Each is decorated with ``enabled=False``: the
module attribute is then a real recorder that never writes, and a test records into its
own store by wrapping ``target.__wrapped__``, which has the same identity. To change a
target's behavior, a test monkeypatches the helper it calls.
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from regression_records import RecordStore, record_func, record_method

NEVER_WRITTEN = Path(tempfile.gettempdir()) / "regression-records-never-written"

#: The store ``inner`` records to. Tests point its root at a temporary directory.
INNER_STORE = RecordStore(NEVER_WRITTEN)


def scale(values):
    return np.asarray(values, dtype=float) * 2.0


def to_int(text):
    return int(text)


@record_func(NEVER_WRITTEN, enabled=False)
def total(values, *, factor=1.0):
    return {"total": float(scale(values).sum() * factor), "n": len(values)}


@record_func(NEVER_WRITTEN, enabled=False)
def parse(text):
    return to_int(text)


@record_func(NEVER_WRITTEN, enabled=False)
def spread(n):
    return {f"k{i}": float(scale([i])[0]) for i in range(1, n + 1)}


@record_func(NEVER_WRITTEN, enabled=False)
def weights(array):
    return scale(array)


#: Enabled, so that it records whenever it is called - except during a replay.
@record_func(INNER_STORE)
def inner(x):
    return x + 1


@record_func(NEVER_WRITTEN, enabled=False)
def nested(x):
    return inner(x) * 10


@record_func(NEVER_WRITTEN, enabled=False)
def threaded(x):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(inner, x).result()


def undecorated(x):
    return x


#: A second name for a recorded function: it resolves to ``total``'s recorder.
alias = total


class Counter:
    def __init__(self):
        self.count = 0

    @record_method(NEVER_WRITTEN, enabled=False)
    def bump(self, n):
        self.count += n
        return self.count
