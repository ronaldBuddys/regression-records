"""One function per thing a replay cannot see, for the gallery page ``15_gotchas``."""

from __future__ import annotations

import enum
from datetime import date
from typing import NamedTuple

from regression_records import record_func, record_method

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/gotchas, or a fresh temporary directory.
STORE = example_store("gotchas")
AUDIT_LOG = STORE.parent / "gotchas-audit.log"

FX_RATES = {"EUR": 1.10, "GBP": 1.25}  # USD per unit, loaded at start-up
MESSAGE = "first wording"
CALLS = {"count": 0}  # what ``tick`` counts
OPEN_ACCOUNTS: set[str] = set()  # filled by ``Account.__init__``, outside any account


class Side(enum.IntEnum):
    """Trade direction; recorded as its integer value, not as a ``Side``."""

    BUY = 1
    SELL = -1


class Quote(NamedTuple):
    """A named tuple, recorded with its class and its fields by name."""

    bid: float
    ask: float


class Point:
    """Plain attributes, recorded straight out of ``__dict__``."""

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y


def today() -> date:
    """Return today's date: the clock the functions below read."""
    return date.today()


def load_prices() -> list[float]:
    """Return the day's prices; the undecorated step upstream of ``average``."""
    return [101.0, 99.5, 100.25]


@record_func(STORE)
def days_to_expiry(expiry: date) -> int:
    """Return the whole days from today until ``expiry``."""
    return (expiry - today()).days


@record_func(STORE)
def to_usd(amount: float, currency: str) -> float:
    """Return ``amount`` of ``currency`` in US dollars, at the loaded rate."""
    return amount * FX_RATES[currency]


@record_func(STORE)
def log_trade(trade_id: str) -> str:
    """Append ``trade_id`` to the audit log, and return the line written."""
    line = f"booked {trade_id}"
    with AUDIT_LOG.open("a") as log:
        log.write(line + "\n")
    return line


@record_func(STORE)
def average(prices: list[float]) -> float:
    """Return the mean of ``prices``."""
    return sum(prices) / len(prices)


@record_func(STORE)
def label(side: Side) -> str:
    """Return the lower-case name of ``side``."""
    return side.name.lower()


@record_func(STORE)
def quote_and_point() -> tuple[Quote, Point]:
    """Return a named tuple and an object, both from an importable module."""
    return Quote(bid=99.5, ask=100.5), Point(1.0, 2.0)


@record_func(STORE)
def fail() -> None:
    """Raise a ``ValueError`` whose message is ``MESSAGE``."""
    raise ValueError(MESSAGE)


@record_func(STORE)
def echo(value: object) -> object:
    """Return ``value`` unchanged."""
    return value


@record_func(STORE)
def tick(step: int) -> int:
    """Add ``step`` to a module-level count, and return the new count."""
    CALLS["count"] += step
    return CALLS["count"]


class Basket:
    """Has a ``size`` property, which ``state_attrs`` cannot select."""

    def __init__(self):
        self.items: list[str] = []

    @property
    def size(self) -> int:
        """Return the number of items."""
        return len(self.items)

    @record_method(STORE, state_attrs=["size"], factory=lambda: Basket())
    def add(self, item: str) -> None:
        """Put ``item`` in the basket."""
        self.items.append(item)


class Account:
    """Registers itself in ``OPEN_ACCOUNTS`` when built: state kept outside itself."""

    def __init__(self, name: str):
        self.name = name
        OPEN_ACCOUNTS.add(name)

    @record_method(STORE)
    def close(self) -> str:
        """Remove this account from ``OPEN_ACCOUNTS``, and return its name."""
        OPEN_ACCOUNTS.remove(self.name)
        return self.name
