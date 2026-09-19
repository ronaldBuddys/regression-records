"""The public decorators: :func:`record_func` and :func:`record_method`."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from ._engine import RecorderConfig, coerce_store, execute
from ._introspect import StateSpec, capture_state, split_receiver
from .comparison.base import ComparatorRegistry
from .comparison.tolerance import ToleranceConfig
from .config import (
    OnMismatch,
    OnNewCase,
    OverwritePolicy,
    RunMode,
    coerce_option,
    resolve_on_new_case,
)
from .identity import FunctionIdentity
from .serialization.base import SerializerRegistry
from .store import RecordStore

F = TypeVar("F", bound=Callable[..., Any])


def record_func(
    store: RecordStore | str | Path,
    *,
    mode: RunMode | str = RunMode.BOTH,
    tolerance: ToleranceConfig | None = None,
    on_new_case: OnNewCase | str | None = None,
    overwrite: OverwritePolicy | str = OverwritePolicy.NEVER,
    on_mismatch: OnMismatch | str = OnMismatch.RAISE,
    enabled: bool = True,
    record_exceptions: bool = True,
    name: str | None = None,
    serializer_registry: SerializerRegistry | None = None,
    comparator_registry: ComparatorRegistry | None = None,
    tags: Mapping[str, str] | None = None,
) -> Callable[[F], F]:
    """Record - and on later runs, regression-test - calls to a plain function.

    On each call the arguments are serialized and hashed; if that exact invocation was
    recorded before, this call's result is compared against it, otherwise it becomes a
    new recorded case (see :class:`~regression_records.config.RunMode` and
    :class:`~regression_records.config.OnNewCase`).

    The wrapped function's observable behavior is unchanged: it receives its original
    arguments untouched, its return value is passed through, and any exception it raises
    is always re-raised. A failed comparison surfaces as
    :class:`~regression_records.exceptions.RegressionMismatchError` (unless
    ``on_mismatch="return"``), and the full
    :class:`~regression_records.comparison.ComparisonResult` is available afterwards on
    the wrapper's ``last_result`` attribute.

    Parameters
    ----------
    store : RecordStore or str or Path
        Where records live - a :class:`RecordStore`, or a path to wrap in one. There is
        no default location; recording somewhere implicit would be worse than being
        asked.
    mode : RunMode or str, default RunMode.BOTH
        Whether this decoration records, validates, or both. Every option accepts the
        enum member or its string value.
    tolerance : ToleranceConfig, optional
        Numeric comparison tolerances; the defaults when omitted.
    on_new_case : OnNewCase or str, optional
        What to do when no record matches these inputs. Defaults to whatever suits
        ``mode``: record the new case, or, under
        :attr:`~regression_records.config.RunMode.VALIDATE_ONLY`, raise
        :class:`~regression_records.exceptions.UnmatchedInvocationError`.
    overwrite : OverwritePolicy or str, default OverwritePolicy.NEVER
        When an existing record may be replaced. Use
        :attr:`~regression_records.config.OverwritePolicy.IF_MISMATCH` for one run to
        accept an intentional behavior change.
    on_mismatch : OnMismatch or str, default OnMismatch.RAISE
        ``"raise"``, or ``"return"`` to return normally and inspect ``last_result``.
    enabled : bool, default True
        When False the wrapper is a pure pass-through, with no store access.
    record_exceptions : bool, default True
        Whether a raised exception is recorded (and compared) in place of a return
        value. When False, a call that raises is neither recorded nor validated - it
        just propagates.
    name : str, optional
        Overrides the qualname used to identify this function on disk. Must be a single
        directory name.
    serializer_registry : SerializerRegistry, optional
        Only used when ``store`` is a path; a ``RecordStore`` passed in directly
        carries its own.
    comparator_registry : ComparatorRegistry, optional
        Comparators used when validating; the default registry when omitted.
    tags : Mapping[str, str], optional
        Free-form metadata stored on each record written by this decoration.

    Returns
    -------
    Callable
        A decorator. The function it returns has three extra attributes:
        ``last_result`` (the :class:`~regression_records.comparison.ComparisonResult`
        of the most recent call, or None if that call recorded, was skipped or raised
        before comparing), ``identity`` and ``config``.

    Raises
    ------
    ValueError
        When the decorator is built: an option is not a member of its enum, or
        ``on_new_case=RECORD`` contradicts ``mode=VALIDATE_ONLY``.
    TypeError
        When the decorator is applied to a generator or async function, or to a
        ``staticmethod`` / ``classmethod`` object.
    RegressionMismatchError
        On a call whose result differs from its record (unless ``on_mismatch="return"``).
    RecordNotFoundError, UnmatchedInvocationError
        On a call with no record to validate against, where recording is not allowed.
    SerializationError
        On a call whose arguments or result cannot be recorded.

    Examples
    --------
    >>> import tempfile
    >>> from regression_records import record_func
    >>> @record_func(tempfile.mkdtemp())
    ... def double(x):
    ...     return x * 2
    >>> double(4)  # first call: recorded
    8
    >>> double(4)  # same arguments: compared against the record
    8
    >>> double.last_result.passed
    True
    """
    return _build_decorator(
        kind="function",
        store=store,
        mode=mode,
        tolerance=tolerance,
        on_new_case=on_new_case,
        overwrite=overwrite,
        on_mismatch=on_mismatch,
        enabled=enabled,
        record_exceptions=record_exceptions,
        name=name,
        serializer_registry=serializer_registry,
        comparator_registry=comparator_registry,
        tags=tags,
    )


def record_method(
    store: RecordStore | str | Path,
    *,
    mode: RunMode | str = RunMode.BOTH,
    tolerance: ToleranceConfig | None = None,
    on_new_case: OnNewCase | str | None = None,
    overwrite: OverwritePolicy | str = OverwritePolicy.NEVER,
    on_mismatch: OnMismatch | str = OnMismatch.RAISE,
    enabled: bool = True,
    record_exceptions: bool = True,
    state_attrs: Sequence[str] | None = None,
    exclude_attrs: Sequence[str] = (),
    name: str | None = None,
    serializer_registry: SerializerRegistry | None = None,
    comparator_registry: ComparatorRegistry | None = None,
    tags: Mapping[str, str] | None = None,
) -> Callable[[F], F]:
    """Record - and on later runs, regression-test - calls to a method.

    Everything :func:`record_func` does, plus the receiver: a method's behavior is as
    much about the object it acts on as about its arguments, so ``self`` is snapshotted
    *before* and *after* the call and both snapshots are recorded and compared. A method
    that mutates its object is therefore regression-tested on the mutation, not only on
    what it returned.

    ``self`` is excluded from the recorded arguments and from the input hash - the state
    describes the object, and hashing the instance itself would make every instance a
    separate case. The wrapper is a plain function assigned on the class, so descriptor
    binding, inheritance, and ``isinstance`` are all unaffected.

    Parameters
    ----------
    store : RecordStore or str or Path
        Where records live, as for :func:`record_func`.
    state_attrs : Sequence[str], optional
        Attribute names to record, or None (default) for every attribute the instance
        exposes. A name found in neither snapshot of a call is an error: it selects
        instance attributes, so a typo or a property would otherwise record nothing.
    exclude_attrs : Sequence[str], default ()
        Attribute names to leave out - the escape hatch for caches, connections, file
        handles, back-references to a parent, and anything else that can't be
        deep-copied or would differ between runs for uninteresting reasons.
    mode, tolerance, on_new_case, overwrite, on_mismatch, enabled, record_exceptions, \
name, serializer_registry, comparator_registry, tags
        Exactly as in :func:`record_func`.

    Returns
    -------
    Callable
        A decorator, whose result carries ``last_result``, ``identity`` and ``config``
        as for :func:`record_func`. ``last_result`` lives on the function, so it is
        shared by every instance of the class.

    Raises
    ------
    TypeError
        When applied to a generator or async function, or above ``@classmethod`` /
        ``@staticmethod`` rather than beneath it. A staticmethod has no receiver: use
        :func:`record_func`.
    SerializationError
        On a call whose receiver exposes no dict-shaped state, whose state cannot be
        deep-copied, or for which ``state_attrs`` names nothing.

    Notes
    -----
    Calls are matched by their arguments alone. Two instances in different states
    making the same call are the same case, and the second is reported as a state
    mismatch rather than recorded separately.
    """
    return _build_decorator(
        kind="method",
        store=store,
        mode=mode,
        tolerance=tolerance,
        on_new_case=on_new_case,
        overwrite=overwrite,
        on_mismatch=on_mismatch,
        enabled=enabled,
        record_exceptions=record_exceptions,
        name=name,
        serializer_registry=serializer_registry,
        comparator_registry=comparator_registry,
        tags=tags,
        state_spec=StateSpec.build(state_attrs, exclude_attrs),
    )


def _build_decorator(
    *,
    kind: str,
    store: RecordStore | str | Path,
    mode: RunMode | str,
    tolerance: ToleranceConfig | None,
    on_new_case: OnNewCase | str | None,
    overwrite: OverwritePolicy | str,
    on_mismatch: OnMismatch | str,
    enabled: bool,
    record_exceptions: bool,
    name: str | None,
    serializer_registry: SerializerRegistry | None,
    comparator_registry: ComparatorRegistry | None,
    tags: Mapping[str, str] | None,
    state_spec: StateSpec | None = None,
) -> Callable[[F], F]:
    # RecorderConfig validates its own fields, and it is built here - at decoration
    # time - so a bad setting fails when the decorator is applied rather than on some
    # later call that happens to reach the relevant branch.
    mode = coerce_option(RunMode, mode, "mode")
    if on_new_case is not None:
        on_new_case = coerce_option(OnNewCase, on_new_case, "on_new_case")
    config = RecorderConfig(
        store=coerce_store(store, serializer_registry),
        mode=mode,
        tolerance=tolerance,
        on_new_case=resolve_on_new_case(mode, on_new_case),
        overwrite=overwrite,
        on_mismatch=on_mismatch,
        record_exceptions=record_exceptions,
        comparator_registry=comparator_registry,
        tags=dict(tags or {}),
    )

    def decorator(func: F) -> F:
        _require_recordable(func, kind)
        identity = FunctionIdentity.from_function(func, kind=kind, name=name)
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return func(*args, **kwargs)

            bound = _bind(signature, args, kwargs)
            if bound is None:
                # The arguments don't fit the signature; let the real call raise the
                # real TypeError rather than reporting it as a recording failure.
                return func(*args, **kwargs)

            if state_spec is None:
                recorded_args, recorded_kwargs = list(bound.args), dict(bound.kwargs)
                snapshot = None
            else:
                instance, recorded_args, recorded_kwargs = split_receiver(bound)
                snapshot = functools.partial(capture_state, instance, state_spec)

            # Cleared first, so that when this call raises - a mismatch, or the
            # function's own exception - last_result is not left describing an earlier,
            # unrelated call.
            wrapper.last_result = None
            outcome = execute(
                config,
                identity,
                lambda: func(*args, **kwargs),
                recorded_args,
                recorded_kwargs,
                capture_state=snapshot,
                check_state=state_spec.check if state_spec is not None else None,
            )
            wrapper.last_result = outcome.comparison
            if outcome.exception is not None:
                raise outcome.exception
            return outcome.value

        wrapper.last_result = None
        wrapper.identity = identity
        wrapper.config = config
        return wrapper  # type: ignore[return-value]

    return decorator


def _require_recordable(func: Any, kind: str) -> None:
    """Reject, at decoration time, callables whose calls cannot be recorded.

    Raises
    ------
    TypeError
        For a generator or async function: calling one returns a generator or coroutine
        object, which is not the function's result and cannot be serialized, so every
        call would fail with a ``SerializationError`` in place of returning it. For a
        ``staticmethod`` or ``classmethod`` object: the decorator was applied above
        ``@staticmethod``/``@classmethod`` rather than beneath it.
    """
    decorator = "record_method" if kind == "method" else "record_func"
    if isinstance(func, staticmethod | classmethod):
        raise TypeError(
            f"{decorator} was applied to a {type(func).__name__} object; apply it "
            f"beneath @{type(func).__name__} so that it wraps the plain function. A "
            "staticmethod has no receiver to snapshot, so record it with record_func."
        )
    target = inspect.unwrap(func)
    if (
        inspect.iscoroutinefunction(target)
        or inspect.isgeneratorfunction(target)
        or inspect.isasyncgenfunction(target)
    ):
        raise TypeError(
            f"{decorator} cannot record '{target.__qualname__}': calling a generator "
            "or async function returns a generator or coroutine object rather than a "
            "result. Record a plain function that consumes it instead."
        )


def _bind(
    signature: inspect.Signature, args: tuple, kwargs: dict
) -> inspect.BoundArguments | None:
    """Normalize a call's arguments so equivalent call styles hash identically.

    ``f(1, 2)``, ``f(1, b=2)`` and ``f(1)`` with ``b`` defaulting to 2 are the same
    invocation and must match the same record, which they only do once bound against
    the signature. Returns None if the arguments don't fit the signature at all.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    return bound
