"""Policy enums controlling what the decorators do on each call.

These are deliberately plain, comparable values rather than behavior-carrying objects:
they are configuration a caller passes in, and the logic that acts on them lives in
:mod:`regression_records.matching` (which case to run) and the decorators (what to do
with the result).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class RunMode(str, Enum):
    """Whether a decorated call may create *new* recorded cases.

    Orthogonal to :class:`OverwritePolicy`, which governs whether an *existing* record
    may be replaced: "validate only" means "do not invent new baselines", not "never
    touch the store" - re-baselining a case you already have is what
    :attr:`OverwritePolicy.IF_MISMATCH` is for, and it is useful precisely in
    ``VALIDATE_ONLY``.
    """

    #: Always write a record; never compare against history.
    RECORD_ONLY = "record_only"
    #: Never record a new case; compare against history, and fail if there is nothing
    #: to compare to.
    VALIDATE_ONLY = "validate_only"
    #: Validate when this exact invocation was recorded before, otherwise record it.
    BOTH = "both"


class OnNewCase(str, Enum):
    """What to do when no historical record matches an invocation's inputs.

    The decorators default to ``None`` rather than to a member of this enum, meaning
    "whatever suits the mode" - see :func:`resolve_on_new_case`.
    """

    #: Treat it as a new case worth keeping and write a record for it.
    RECORD = "record"
    #: Refuse to proceed silently - raise ``UnmatchedInvocationError``.
    ERROR = "error"
    #: Call through and neither record nor validate.
    IGNORE = "ignore"


def resolve_on_new_case(mode: RunMode, on_new_case: Optional[OnNewCase]) -> OnNewCase:
    """Resolve the ``on_new_case=None`` default against ``mode``.

    ``None`` means mode-appropriate: :attr:`OnNewCase.RECORD` where recording is
    allowed, :attr:`OnNewCase.ERROR` under :attr:`RunMode.VALIDATE_ONLY`, where an
    input nothing was recorded for is exactly what you want surfaced.

    Asking *explicitly* for :attr:`OnNewCase.RECORD` under ``VALIDATE_ONLY`` is a
    genuine contradiction - one setting forbids writing, the other requires it - and
    raises here. Callers resolve at decoration time, so the contradiction is reported
    when the decorator is applied rather than weeks later on the first call that
    happens to reach novel inputs.
    """
    if on_new_case is None:
        return OnNewCase.ERROR if mode is RunMode.VALIDATE_ONLY else OnNewCase.RECORD
    if mode is RunMode.VALIDATE_ONLY and on_new_case is OnNewCase.RECORD:
        raise ValueError(
            "on_new_case=OnNewCase.RECORD contradicts mode=RunMode.VALIDATE_ONLY: one "
            "asks for new cases to be recorded, the other forbids writing. Use "
            "RunMode.BOTH to record new cases as they appear, or leave on_new_case "
            "unset to error on them."
        )
    return on_new_case


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
