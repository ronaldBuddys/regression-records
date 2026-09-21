"""Comparing a recorded exception against what a fresh call actually raised.

This lives in the comparison layer rather than in the engine because *how a difference
is worded* is this package's job - the engine decides only which values to compare. Both
sides are described by exception type name alone, so nothing here needs the record model.
That name is also how a record stores the exception, so :func:`exception_type_name`
lives here too.
"""

from __future__ import annotations

from .base import ComparisonResult, Diff
from .paths import OUTPUT_ROOT

#: Stands in for an exception type on whichever side the call returned a value instead.
RETURNED_NORMALLY = "<returned normally>"


def exception_type_name(exc: BaseException) -> str:
    """Return the module-qualified name of ``exc``'s type, as records store it.

    Builtins keep their ``builtins.`` prefix, so every name this returns contains a dot.
    A bare ``__name__`` - what schema version 1 records hold - never does, which is how
    :func:`compare_exceptions` tells the two apart.

    Parameters
    ----------
    exc : BaseException
        The exception a call raised.

    Returns
    -------
    str
        ``"<module>.<qualname>"``, e.g. ``"builtins.ValueError"`` or
        ``"requests.exceptions.ConnectionError"``.
    """
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def compare_exceptions(
    expected_type: str | None,
    actual: BaseException | None,
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

    ``expected_type`` is module-qualified (see :func:`exception_type_name`), so two
    same-named exceptions from different modules differ. A schema version 1 record
    holds the bare class name instead; that is compared by bare name, and two
    same-named types still match against it.
    """
    actual_type = None
    if actual is not None:
        # No dot means a v1 bare name: compare like with like.
        bare = expected_type is not None and "." not in expected_type
        actual_type = type(actual).__name__ if bare else exception_type_name(actual)
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
