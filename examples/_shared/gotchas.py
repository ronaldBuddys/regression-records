"""The functions example 13 records and replays, one per thing replay cannot see.

Example 13 changes the module-level state here - ``today``, ``FX_RATES``,
``load_prices``, ``MESSAGE`` - to show what each function's replay depends on, and
restores it afterwards.
"""

from __future__ import annotations

import enum
from datetime import date

from regression_records import record_func

from .demo import EXAMPLE_STORES
from .models import Point, Quote

STORE = EXAMPLE_STORES / "13_gotchas"
AUDIT_LOG = EXAMPLE_STORES / "13_gotchas_audit.log"

FX_RATES = {"EUR": 1.10, "GBP": 1.25}  # USD per unit, loaded at start-up
MESSAGE = "first wording"


class Side(enum.IntEnum):
    """Trade direction; recorded as its integer value, not as a ``Side``."""

    BUY = 1
    SELL = -1


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
