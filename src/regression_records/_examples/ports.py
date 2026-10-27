"""A port parser that raises, for the gallery page ``08_exceptions_and_drift``."""

from __future__ import annotations

from regression_records import record_func

from . import example_store

# $REGRESSION_RECORDS_EXAMPLE_STORE/ports, or a fresh temporary directory.
STORE = example_store("ports")

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


@record_func(STORE, record_exceptions=False)
def open_port(port: int) -> None:
    """Fail to connect, as a flaky network call would; never recorded when it raises."""
    raise ConnectionError(f"nothing listening on port {port}")
