"""Policy enums controlling what the decorators do on each call.

These are deliberately plain, comparable values rather than behavior-carrying objects:
they are configuration a caller passes in, and the logic that acts on them lives in
:mod:`regression_records.matching` (which case to run) and the decorators (what to do
with the result).
"""

from __future__ import annotations

from enum import Enum


class RunMode(str, Enum):
    """What a decorated call does with the record store."""

    #: Always write a record; never compare against history.
    RECORD_ONLY = "record_only"
    #: Never write; compare against history and fail if there is nothing to compare to.
    VALIDATE_ONLY = "validate_only"
    #: Validate when this exact invocation was recorded before, otherwise record it.
    BOTH = "both"


class OnNewCase(str, Enum):
    """What to do when no historical record matches an invocation's inputs."""

    #: Treat it as a new case worth keeping and write a record for it.
    RECORD = "record"
    #: Refuse to proceed silently - raise ``UnmatchedInvocationError``.
    ERROR = "error"
    #: Call through and neither record nor validate.
    IGNORE = "ignore"


class OnMismatch(str, Enum):
    """What a decorated call does when a comparison against history fails."""

    #: Raise ``RegressionMismatchError``.
    RAISE = "raise"
    #: Return normally; the caller inspects the wrapper's ``last_result``.
    RETURN = "return"


class OverwritePolicy(str, Enum):
    """When an existing record for the same inputs may be replaced on disk."""

    #: Keep history immutable; a mismatch never touches the store.
    NEVER = "never"
    #: Replace the matched record when the new result differs - the workflow for
    #: accepting an intentional behavior change.
    IF_MISMATCH = "if_mismatch"
    #: Rewrite the matched record on every call.
    ALWAYS = "always"
