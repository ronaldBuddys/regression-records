"""Policy enums controlling what the decorators do on each call.

These are deliberately plain, comparable values rather than behavior-carrying objects:
they are configuration a caller passes in, and the logic that acts on them lives in
:mod:`regression_records.matching` (which case to run) and the decorators (what to do
with the result).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

E = TypeVar("E", bound=StrEnum)


class RunMode(StrEnum):
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


class OnNewCase(StrEnum):
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


def coerce_option(enum_cls: type[E], value: E | str, name: str) -> E:
    """Return ``value`` as a member of ``enum_cls``, accepting the member's string value.

    Every option is normalized through here before anything reads it, because the
    engine dispatches on identity (``mode is RunMode.VALIDATE_ONLY``). A plain string
    equals its member but is not identical to it, so an uncoerced
    ``mode="validate_only"`` would match no branch and silently behave as the default
    - recording where the caller asked for validation only.

    Parameters
    ----------
    enum_cls : type[StrEnum]
        The option's enum, e.g. :class:`RunMode`.
    value : StrEnum or str
        A member of ``enum_cls``, or a member's string value (``"validate_only"``).
        Matching is exact: case and surrounding whitespace count.
    name : str
        The option's name, used only in the error message.

    Returns
    -------
    StrEnum
        The member of ``enum_cls`` itself, safe to test with ``is``.

    Raises
    ------
    ValueError
        If ``value`` is neither a member of ``enum_cls`` nor one of its values.
    """
    try:
        return enum_cls(value)
    except ValueError:
        raise ValueError(
            f"{name} must be one of {[member.value for member in enum_cls]}, "
            f"got {value!r}"
        ) from None


def resolve_on_new_case(mode: RunMode, on_new_case: OnNewCase | None) -> OnNewCase:
    """Resolve the ``on_new_case=None`` default against ``mode``.

    ``None`` means mode-appropriate: :attr:`OnNewCase.RECORD` where recording is
    allowed, :attr:`OnNewCase.ERROR` under :attr:`RunMode.VALIDATE_ONLY`, where an
    input nothing was recorded for is exactly what you want surfaced.

    Asking *explicitly* for :attr:`OnNewCase.RECORD` under ``VALIDATE_ONLY`` is a
    genuine contradiction - one setting forbids writing, the other requires it - and
    raises here. Callers resolve at decoration time, so the contradiction is reported
    when the decorator is applied rather than weeks later on the first call that
    happens to reach novel inputs.

    Parameters
    ----------
    mode : RunMode
        The run mode, already coerced to an enum member (see :func:`coerce_option`).
    on_new_case : OnNewCase, optional
        The caller's choice, already coerced, or None for the mode-appropriate default.

    Returns
    -------
    OnNewCase
        ``on_new_case`` when given; otherwise ``ERROR`` under ``VALIDATE_ONLY`` and
        ``RECORD`` under the other modes. Never None.

    Raises
    ------
    ValueError
        If ``on_new_case`` is ``RECORD`` and ``mode`` is ``VALIDATE_ONLY``.
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


class OnMismatch(StrEnum):
    """What a decorated call does when a comparison against history fails."""

    #: Raise ``RegressionMismatchError``.
    RAISE = "raise"
    #: Return normally; the caller inspects the wrapper's ``last_result``.
    RETURN = "return"


class OverwritePolicy(StrEnum):
    """When an existing record for the same inputs may be replaced on disk."""

    #: Keep history immutable; a mismatch never touches the store.
    NEVER = "never"
    #: Replace the matched record when the new result differs - the workflow for
    #: accepting an intentional behavior change.
    IF_MISMATCH = "if_mismatch"
    #: Rewrite the matched record on every call.
    ALWAYS = "always"
