"""Exception types raised by regression-records."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .comparison.base import ComparisonResult


class RegressionRecordsError(Exception):
    """Base class for all exceptions raised by regression-records."""


class SerializationError(RegressionRecordsError):
    """Raised when a value cannot be converted to the storable representation."""


class DeserializationError(RegressionRecordsError):
    """Raised when a stored value cannot be reconstructed."""


class CorruptRecordError(DeserializationError):
    """Raised when a record's JSON, or a file it references, is missing or malformed."""


class RecordNotFoundError(RegressionRecordsError):
    """Raised in validate-only mode when no historical records exist for an identity."""


class UnmatchedInvocationError(RegressionRecordsError):
    """Raised when ``on_new_case=ERROR`` and no historical record matches the call."""


class RegressionMismatchError(RegressionRecordsError):
    """Raised when a new execution's result does not match its matched historical record."""

    def __init__(self, result: ComparisonResult) -> None:
        self.result = result
        super().__init__(result.summary())


class IdentityResolutionError(RegressionRecordsError):
    """Raised when a function, method, or class cannot be re-imported by name."""


class CodeDriftWarning(UserWarning):
    """Warned when validating against a record captured from different source code.

    Not an error: a record stays valid across refactors, and the whole point of a
    regression suite is to check that changed code still behaves the same. It is worth
    surfacing, though, because it tells you whether a passing comparison confirms
    unchanged behavior across a real change, or merely that nothing changed at all.
    """
