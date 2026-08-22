"""Comparing a recorded exception against what a fresh call actually raised.

This lives in the comparison layer rather than in the engine because *how a difference
is worded* is this package's job - the engine decides only which values to compare. Both
sides are described by exception type name alone, so nothing here needs the record model.
"""

from __future__ import annotations

from typing import Optional

from .base import ComparisonResult, Diff
from .paths import OUTPUT_ROOT

#: Stands in for an exception type on whichever side the call returned a value instead.
RETURNED_NORMALLY = "<returned normally>"


def compare_exceptions(
    expected_type: Optional[str],
    actual: Optional[BaseException],
    *,
    root: str = OUTPUT_ROOT,
) -> ComparisonResult:
    """Compare what a recorded call raised against what this call raised.

    ``None`` on either side means that call returned a value instead of raising, so
    exception-vs-success in either direction is always a difference.

    Only the exception *type* is compared: messages routinely carry nondeterministic
    detail (paths, addresses, timings) that would turn every run into a false mismatch.
    Compare messages by registering a ``Comparator`` for your exception type, or by
    normalizing the message in the wrapped callable.
    """
    actual_type = type(actual).__name__ if actual is not None else None
    if expected_type == actual_type:
        return ComparisonResult(passed=True)

    if expected_type is None:
        message = (
            f"the recorded call returned a value, but this call raised "
            f"{actual_type}: {actual}"
        )
    elif actual_type is None:
        message = (
            f"the recorded call raised {expected_type}, but this call returned normally"
        )
    else:
        message = f"raised {actual_type}, expected {expected_type}"

    return ComparisonResult.from_diffs(
        [
            Diff(
                path=root,
                kind="exception_mismatch",
                expected=expected_type or RETURNED_NORMALLY,
                actual=actual_type or RETURNED_NORMALLY,
                message=message,
            )
        ]
    )
