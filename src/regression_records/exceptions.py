"""Exception types raised by regression-records."""

from __future__ import annotations


class RegressionRecordsError(Exception):
    """Base class for all exceptions raised by regression-records."""


class SerializationError(RegressionRecordsError):
    """Raised when a value cannot be converted to the storable representation."""


class DeserializationError(RegressionRecordsError):
    """Raised when a stored value cannot be reconstructed."""


class CorruptRecordError(DeserializationError):
    """Raised when a record's JSON, or a file it references, is missing or malformed."""


class StoreNotConfiguredError(RegressionRecordsError, ValueError):
    """Raised when no store was passed and ``REGRESSION_RECORDS_STORE`` is not set."""


class IdentityResolutionError(RegressionRecordsError):
    """Raised when a function, method, or class cannot be re-imported by name."""


class CodeDriftWarning(UserWarning):
    """Warned when a record passes replay but its function's source has changed.

    :func:`~regression_records.replay_tests` warns it; :func:`~regression_records.replay`
    reports the same thing as :attr:`ReplayCase.code_drift
    <regression_records.replay.ReplayCase.code_drift>`.

    Not an error: a record stays valid across refactors, and the whole point of a
    regression suite is to check that changed code still behaves the same. It is worth
    surfacing, though, because it tells you whether a pass confirms unchanged behavior
    across a real change, or merely that nothing changed at all.
    """
