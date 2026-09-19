"""The public decorators: :func:`record_func` and :func:`record_method`."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TypeVar

from ._engine import RecorderConfig, coerce_store, execute
from ._introspect import StateSpec, capture_state, split_receiver
from .comparison.base import ComparatorRegistry
from .comparison.tolerance import ToleranceConfig
from .config import OnMismatch, OnNewCase, OverwritePolicy, RunMode, resolve_on_new_case
from .identity import FunctionIdentity
from .serialization.base import SerializerRegistry
from .store import RecordStore

F = TypeVar("F", bound=Callable[..., Any])


def record_func(
    store: RecordStore | str | Path,
    *,
    mode: RunMode = RunMode.BOTH,
    tolerance: Optional[ToleranceConfig] = None,
    on_new_case: Optional[OnNewCase] = None,
    overwrite: OverwritePolicy = OverwritePolicy.NEVER,
    on_mismatch: OnMismatch = OnMismatch.RAISE,
    enabled: bool = True,
    record_exceptions: bool = True,
    name: Optional[str] = None,
    serializer_registry: Optional[SerializerRegistry] = None,
    comparator_registry: Optional[ComparatorRegistry] = None,
    tags: Optional[Mapping[str, str]] = None,
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

    :param store: where records live - a :class:`RecordStore`, or a path to wrap in one.
        There is no default location; recording somewhere implicit would be worse than
        being asked.
    :param mode: whether this decoration records, validates, or both.
    :param tolerance: numeric comparison tolerances; see :class:`ToleranceConfig`.
    :param on_new_case: what to do when no record matches these inputs. Defaults to
        whatever suits ``mode`` - record the new case, or, under
        :attr:`~regression_records.config.RunMode.VALIDATE_ONLY`, raise
        :class:`~regression_records.exceptions.UnmatchedInvocationError`.
    :param overwrite: when an existing record may be replaced - use
        :attr:`~regression_records.config.OverwritePolicy.IF_MISMATCH` to accept an
        intentional behavior change.
    :param on_mismatch: ``"raise"``, or ``"return"`` to inspect ``last_result`` instead.
    :param enabled: when False the wrapper is a pure pass-through, with no store access.
    :param record_exceptions: whether a raised exception is recorded (and compared) in
        place of a return value. When False, a call that raises is neither recorded nor
        validated - it just propagates.
    :param name: overrides the qualname used to identify this function on disk.
    :param serializer_registry: only used when ``store`` is a path; a ``RecordStore``
        passed in directly carries its own.
    :param comparator_registry: comparators used when validating.
    :param tags: free-form metadata stored on each record written by this decoration.
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
    mode: RunMode = RunMode.BOTH,
    tolerance: Optional[ToleranceConfig] = None,
    on_new_case: Optional[OnNewCase] = None,
    overwrite: OverwritePolicy = OverwritePolicy.NEVER,
    on_mismatch: OnMismatch = OnMismatch.RAISE,
    enabled: bool = True,
    record_exceptions: bool = True,
    state_attrs: Optional[Sequence[str]] = None,
    exclude_attrs: Sequence[str] = (),
    name: Optional[str] = None,
    serializer_registry: Optional[SerializerRegistry] = None,
    comparator_registry: Optional[ComparatorRegistry] = None,
    tags: Optional[Mapping[str, str]] = None,
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

    :param state_attrs: attribute names to record, or None (default) for every attribute
        the instance exposes.
    :param exclude_attrs: attribute names to leave out - the escape hatch for caches,
        connections, file handles, and anything else that can't be deep-copied or would
        differ between runs for uninteresting reasons.

    All other parameters behave exactly as in :func:`record_func`.
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
    mode: RunMode,
    tolerance: Optional[ToleranceConfig],
    on_new_case: Optional[OnNewCase],
    overwrite: OverwritePolicy,
    on_mismatch: OnMismatch,
    enabled: bool,
    record_exceptions: bool,
    name: Optional[str],
    serializer_registry: Optional[SerializerRegistry],
    comparator_registry: Optional[ComparatorRegistry],
    tags: Optional[Mapping[str, str]],
    state_spec: Optional[StateSpec] = None,
) -> Callable[[F], F]:
    # RecorderConfig validates its own fields, and it is built here - at decoration
    # time - so a bad setting fails when the decorator is applied rather than on some
    # later call that happens to reach the relevant branch.
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

            outcome = execute(
                config,
                identity,
                lambda: func(*args, **kwargs),
                recorded_args,
                recorded_kwargs,
                capture_state=snapshot,
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


def _bind(
    signature: inspect.Signature, args: tuple, kwargs: dict
) -> Optional[inspect.BoundArguments]:
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
