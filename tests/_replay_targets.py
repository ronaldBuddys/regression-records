"""Importable recorded functions for the replay tests.

Replay imports a function by its recorded module and qualname, so these must live at
module level of an importable module. Each is decorated with ``enabled=False``: the
module attribute is then a real recorder that never writes, and a test records into its
own store by wrapping ``target.__wrapped__``, which has the same identity. To change a
target's behavior, a test monkeypatches the helper it calls.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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


#: Whether ``Widget.resize`` keeps its history; a test switches it off.
KEEP_HISTORY = True


class Widget:
    def __init__(self, width):
        self.width = width
        self.history = []

    @record_method(NEVER_WRITTEN, enabled=False)
    def resize(self, width):
        if KEEP_HISTORY:
            self.history.append(self.width)
        self.width = width
        return width


class Connection:
    """Stands in for a live resource: a recorded copy of one is no longer live."""

    def __init__(self):
        self.live = True

    def __deepcopy__(self, memo):
        stale = Connection()
        stale.live = False
        return stale


def new_account():
    return Account(Connection())


class Account:
    """Records only ``balance``; replay builds the rest with ``new_account``."""

    def __init__(self, connection):
        self.connection = connection
        self.balance = 0.0

    @record_method(
        NEVER_WRITTEN, exclude_attrs=["connection"], factory=new_account, enabled=False
    )
    def deposit(self, amount):
        if not self.connection.live:
            raise RuntimeError("the connection is not live")
        self.balance += amount
        return self.balance


class Wrong:
    pass


class Misbuilt(Account):
    """Its factory builds the wrong kind of object."""

    @record_method(
        NEVER_WRITTEN, exclude_attrs=["connection"], factory=Wrong, enabled=False
    )
    def deposit(self, amount):
        self.balance += amount
        return self.balance


class Stateful:
    """Defines its own state protocol, which replay must use to rebuild it."""

    def __init__(self, value):
        self._value = value

    def __getstate__(self):
        return {"value": self._value}

    def __setstate__(self, state):
        self._value = state["value"]

    @record_method(NEVER_WRITTEN, enabled=False)
    def double(self):
        self._value *= 2
        return self._value


class SlottedCounter:
    __slots__ = ("count",)

    def __init__(self):
        self.count = 0

    @record_method(NEVER_WRITTEN, enabled=False)
    def bump(self, n):
        self.count += n
        return self.count


@dataclass(frozen=True, slots=True)
class FrozenPoint:
    x: float
    y: float

    @record_method(NEVER_WRITTEN, enabled=False)
    def norm(self):
        return float(np.hypot(self.x, self.y))


@record_func(NEVER_WRITTEN, enabled=False)
def spread_of(quote):
    return quote.ask - quote.bid


#: The variables the environment targets read.
CURRENCY = "RR_TEST_CURRENCY"
API_KEY = "RR_TEST_API_KEY"


@record_func(NEVER_WRITTEN, enabled=False)
def in_currency(amount):
    """Read a setting from the environment during the call."""
    return f"{amount} {os.environ.get(CURRENCY, 'unset')}"


def new_wallet():
    """A factory that reads the environment too."""
    wallet = Wallet()
    wallet.currency = os.environ.get(CURRENCY, "unset")
    return wallet


class Wallet:
    def __init__(self):
        self.currency = "unset"
        self.total = 0.0

    @record_method(
        NEVER_WRITTEN, exclude_attrs=["currency"], factory=new_wallet, enabled=False
    )
    def add(self, amount):
        self.total += amount
        return f"{self.total} {self.currency}"
