"""Whether a decorated call may record, and the marker replay uses to find a recorder.

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
