"""Whether a decorated call records, and the marker replay uses to find a recorder.

Recording is off unless the environment variable ``REGRESSION_RECORDS`` switches it on,
read afresh on every call so a test can flip it. An environment variable rather than a
module global, because it reaches subprocesses and CI jobs without code changes.

Replay reruns recorded functions, and those functions may themselves call decorated
functions. None of them may write while a replay is running, whatever the environment
says: a replay observes, and a record written during one would capture current behavior
as if it were a baseline.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._introspect import StateSpec
    from .identity import FunctionIdentity, Kind

#: The environment variable that switches recording on.
ENV_VAR = "REGRESSION_RECORDS"

#: The attribute a decorated wrapper carries, holding its :class:`RecorderInfo`.
MARKER = "__regression_records__"

_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"", "0", "false", "no", "off"})

_lock = threading.Lock()
_suppressed = 0


@dataclass(frozen=True)
class RecorderInfo:
    """What replay needs to know about a decorated callable.

    Attributes
    ----------
    identity : FunctionIdentity
        The identity its records are stored under.
    kind : {"function", "method"}
        Which decorator produced it.
    state_spec : StateSpec or None
        The receiver attributes a method records; None for a function.
    """

    identity: FunctionIdentity
    kind: Kind
    state_spec: StateSpec | None = None


def is_suppressed() -> bool:
    """Return whether a replay is running, in which case nothing may be recorded."""
    return _suppressed > 0


@contextmanager
def suppressed() -> Iterator[None]:
    """Forbid recording for the duration, in every thread and in child processes.

    A process-wide counter rather than a context variable, so that a decorated call a
    replayed function makes from a worker thread is suppressed too. The environment
    variable is forced off as well, for subprocesses the function starts; its previous
    value is restored on exit.
    """
    global _suppressed
    with _lock:
        _suppressed += 1
        previous = os.environ.get(ENV_VAR)
        os.environ[ENV_VAR] = "0"
    try:
        yield
    finally:
        with _lock:
            _suppressed -= 1
            if previous is None:
                os.environ.pop(ENV_VAR, None)
            else:
                os.environ[ENV_VAR] = previous


def recording_enabled() -> bool:
    """Return whether decorated calls record right now.

    Returns
    -------
    bool
        False while a replay runs. Otherwise whether ``REGRESSION_RECORDS`` is set to
        ``1``, ``true``, ``yes`` or ``on`` (any case); unset, empty, ``0``, ``false``,
        ``no`` and ``off`` mean off.

    Raises
    ------
    ValueError
        If ``REGRESSION_RECORDS`` holds anything else: a typo must not silently leave
        recording off.
    """
    if is_suppressed():
        return False
    value = os.environ.get(ENV_VAR, "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    raise ValueError(
        f"{ENV_VAR}={os.environ[ENV_VAR]!r} is not a recognised switch; use 1 to "
        "record, or 0 (or leave it unset) not to"
    )


def enable() -> None:
    """Switch recording on for this process and the subprocesses it starts.

    Sets ``REGRESSION_RECORDS=1``.
    """
    os.environ[ENV_VAR] = "1"


def disable() -> None:
    """Switch recording off for this process and the subprocesses it starts.

    Removes ``REGRESSION_RECORDS``.
    """
    os.environ.pop(ENV_VAR, None)


@contextmanager
def recording(enabled: bool = True) -> Iterator[None]:
    """Switch recording on (or off) for the duration of a ``with`` block.

    Parameters
    ----------
    enabled : bool, default True
        Whether calls inside the block record.

    Examples
    --------
    >>> from regression_records import recording
    >>> with recording():
    ...     run_pipeline()

    Notes
    -----
    This sets the environment variable, which is process-wide: another thread
    recording at the same time sees the change too.
    """
    previous = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = previous
