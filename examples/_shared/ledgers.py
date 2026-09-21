"""The classes example 6 records and replays: methods, and the objects they act on.

``KEEP_HISTORY`` lets example 6 break ``Ledger.apply`` without editing source: switched
off, ``apply`` still returns the right balance but stops appending to ``history``.
Example 6 switches it off for one replay and restores it.
"""

from __future__ import annotations

import threading

from regression_records import record_method

from .demo import EXAMPLE_STORES
from .models import Point

STORE = EXAMPLE_STORES / "06_recording_methods"

KEEP_HISTORY = True
PRICES = {"ACME": 101.5, "INITECH": 42.0}


class Ledger:
    """A balance and the balances it held before each change."""

    def __init__(self, balance: float):
        self.balance = balance
        self.history: list[float] = []

    @record_method(STORE)
    def apply(self, delta: float) -> float:
        """Add ``delta`` to the balance, keeping the old balance in ``history``."""
        if KEEP_HISTORY:
            self.history.append(self.balance)
        self.balance += delta
        return self.balance


class Connection:
    """Stands in for a database connection: live, and impossible to deep-copy."""

    def __init__(self):
        self.lock = threading.Lock()

    def lookup(self, symbol: str) -> float:
        """Return the price of ``symbol``."""
        return PRICES[symbol]


class NaivePricer:
    """Records every attribute, connection included, which cannot be snapshotted."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self.quotes = 0

    @record_method(STORE)
    def value(self, symbol: str, quantity: int) -> float:
        """Return the value of ``quantity`` units of ``symbol``."""
        self.quotes += 1
        return quantity * self.connection.lookup(symbol)


def new_pricer() -> Pricer:
    """Build a ``Pricer`` on a fresh connection: the factory replay rebuilds it with."""
    return Pricer(Connection())


class Pricer:
    """Leaves its connection out of the record; replay gets one from ``new_pricer``."""

    def __init__(self, connection: Connection):
        self.connection = connection
        self.quotes = 0

    @record_method(STORE, exclude_attrs=("connection",), factory=new_pricer)
    def value(self, symbol: str, quantity: int) -> float:
        """Return the value of ``quantity`` units of ``symbol``."""
        self.quotes += 1
        return quantity * self.connection.lookup(symbol)


class Survey:
    """Records only the two attributes named in ``state_attrs``."""

    def __init__(self):
        self.origin = Point(0.0, 0.0)
        self.marks: list[Point] = []

    @record_method(STORE, state_attrs=("origin", "marks"), factory=lambda: Survey())
    def mark(self, x: float, y: float) -> int:
        """Add a mark at ``(x, y)``, move the origin halfway to it, return the count."""
        self.marks.append(Point(x, y))
        self.origin = Point(x / 2, y / 2)
        return len(self.marks)
