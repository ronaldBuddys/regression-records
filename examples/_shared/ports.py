"""The function example 7 records: a parser that rejects bad input.

The module-level settings let example 7 change its behaviour without editing source:
``STRICT`` turns the range check off, ``RANGE_ERROR`` changes the exception it raises
and ``MAX_PORT`` narrows the range.
"""

from __future__ import annotations

from regression_records import record_func

from .demo import EXAMPLE_STORES

STORE = EXAMPLE_STORES / "07_exceptions_and_drift"

STRICT = True
RANGE_ERROR: type[Exception] = ValueError
MAX_PORT = 65535


@record_func(STORE)
def parse_port(text: str) -> int:
    """Return ``text`` as a TCP port number, rejecting one out of range."""
    port = int(text)
    if STRICT and not 0 < port <= MAX_PORT:
        raise RANGE_ERROR(f"port out of range: {port}")
    return port
