"""regression-records: record and regression-test Python function/method executions."""

from __future__ import annotations

from ._version import __version__
from .exceptions import (
    CorruptRecordError,
    DeserializationError,
    IdentityResolutionError,
    RecordNotFoundError,
    RegressionMismatchError,
    RegressionRecordsError,
    SerializationError,
    UnmatchedInvocationError,
)

__all__ = [
    "CorruptRecordError",
    "DeserializationError",
    "IdentityResolutionError",
    "RecordNotFoundError",
    "RegressionMismatchError",
    "RegressionRecordsError",
    "SerializationError",
    "UnmatchedInvocationError",
    "__version__",
]
