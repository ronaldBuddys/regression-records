"""The public decorators: :func:`record_func` and :func:`record_method`."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from ._engine import RecorderConfig, record_call
from ._introspect import StateSpec, capture_state, split_receiver
from ._switch import MARKER, RecorderInfo, recording_enabled
from .identity import FunctionIdentity, Kind
from .serialization.base import SerializerRegistry
from .store import RecordStore, coerce_store

F = TypeVar("F", bound=Callable[..., Any])


def record_func(
    store: RecordStore | str | Path,
    *,
    record_exceptions: bool = True,
    tags: Mapping[str, str] | None = None,
    serializer_registry: SerializerRegistry | None = None,
    enabled: bool = True,
) -> Callable[[F], F]:
    """Record calls to a plain function, when recording is switched on.

    Recording is off unless the environment variable ``REGRESSION_RECORDS`` is set to
    ``1`` (see :func:`~regression_records.enable` and
    :func:`~regression_records.recording`); until then the wrapper is a pass-through
    costing one check. With it on, each call's normalized arguments and what it returned
    or raised are written to ``store`` as a record, for
    :func:`~regression_records.replay` to check later.

    The wrapped function's observable behavior is unchanged: it receives its original
    arguments untouched, its return value is passed through, and any exception it raises
    is re-raised.

    Records are never replaced. Recording a call that behaves exactly as an earlier
    recording writes nothing; recording one that behaves differently adds a record
    beside the old one, which then fails replay until you delete it.

    Parameters
    ----------
    store : RecordStore or str or Path
        Where records are written - a :class:`RecordStore`, or a path to wrap in one.
        There is no default location; recording somewhere implicit would be worse than
        being asked.
    record_exceptions : bool, default True
        Whether a call that raises is recorded, with the exception in place of a return
        value. When False it just propagates.
    tags : Mapping[str, str], optional
        Free-form metadata stored on each record written by this decoration.
    serializer_registry : SerializerRegistry, optional
        Only used when ``store`` is a path; a ``RecordStore`` passed in directly
        carries its own.
    enabled : bool, default True
        When False the wrapper never records, whatever the environment says.

    Returns
    -------
    Callable
        A decorator. The function it returns has an ``identity`` attribute: the
        :class:`~regression_records.identity.FunctionIdentity` its records are filed
        under, and replay imports it by.

    Raises
    ------
    TypeError
        When the decorator is applied to a generator or async function, or to a
        ``staticmethod`` / ``classmethod`` object.
    SerializationError
        On a recorded call whose arguments or result cannot be recorded.
    ValueError
        On any call, if ``REGRESSION_RECORDS`` holds an unrecognised value.

    Examples
    --------
    >>> import tempfile
    >>> from regression_records import record_func, recording
    >>> @record_func(tempfile.mkdtemp())
    ... def double(x):
    ...     return x * 2
    >>> double(4)  # recording is off: a plain call
    8
    >>> with recording():
    ...     double(4)  # recorded
    8
    """
    return _build_decorator(
        kind="function",
        store=store,
        enabled=enabled,
        record_exceptions=record_exceptions,
        serializer_registry=serializer_registry,
        tags=tags,
    )


def record_method(
    store: RecordStore | str | Path,
    *,
    state_attrs: Sequence[str] | None = None,
    exclude_attrs: Sequence[str] = (),
    record_exceptions: bool = True,
    tags: Mapping[str, str] | None = None,
    serializer_registry: SerializerRegistry | None = None,
    enabled: bool = True,
) -> Callable[[F], F]:
    """Record calls to a method, with the object's state before and after.

    Everything :func:`record_func` does, plus the receiver: a method's behavior is as
    much about the object it acts on as about its arguments, so ``self`` is snapshotted
    *before* and *after* the call and both snapshots are recorded. A method that mutates
    its object is therefore checked on the mutation, not only on what it returned.

    ``self`` is excluded from the recorded arguments and from the input hash - the state
    describes the object, and hashing the instance itself would make every instance a
    separate case. The wrapper is a plain function assigned on the class, so descriptor
    binding, inheritance, and ``isinstance`` are all unaffected.

    Parameters
    ----------
    store : RecordStore or str or Path
        Where records are written, as for :func:`record_func`.
    state_attrs : Sequence[str], optional
        Attribute names to record, or None (default) for every attribute the instance
        exposes. A name found in neither snapshot of a call is an error: it selects
        instance attributes, so a typo or a property would otherwise record nothing.
    exclude_attrs : Sequence[str], default ()
        Attribute names to leave out - the escape hatch for caches, connections, file
        handles, back-references to a parent, and anything else that can't be
        deep-copied or would differ between runs for uninteresting reasons.
    record_exceptions, tags, serializer_registry, enabled
        Exactly as in :func:`record_func`.

    Returns
    -------
    Callable
        A decorator, whose result carries ``identity`` as for :func:`record_func`.

    Raises
    ------
    TypeError
        When applied to a generator or async function, or above ``@classmethod`` /
        ``@staticmethod`` rather than beneath it. A staticmethod has no receiver: use
        :func:`record_func`.
    SerializationError
        On a recorded call whose receiver exposes no dict-shaped state, whose state
        cannot be deep-copied, or for which ``state_attrs`` names nothing.
    """
    return _build_decorator(
        kind="method",
        store=store,
        enabled=enabled,
        record_exceptions=record_exceptions,
        serializer_registry=serializer_registry,
        tags=tags,
        state_spec=StateSpec.build(state_attrs, exclude_attrs),
    )


def _build_decorator(
    *,
    kind: Kind,
    store: RecordStore | str | Path,
    enabled: bool,
    record_exceptions: bool,
    serializer_registry: SerializerRegistry | None,
    tags: Mapping[str, str] | None,
    state_spec: StateSpec | None = None,
) -> Callable[[F], F]:
    config = RecorderConfig(
        store=coerce_store(store, serializer_registry),
        record_exceptions=record_exceptions,
        tags=dict(tags or {}),
    )

    def decorator(func: F) -> F:
        _require_recordable(func, kind)
        identity = FunctionIdentity.from_function(func, kind=kind)
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Read on every call, so a test (or a with-block) can switch it; while a
            # replay runs it is always off, so replaying never writes.
            if not enabled or not recording_enabled():
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

            outcome = record_call(
                config,
                identity,
                lambda: func(*args, **kwargs),
                recorded_args,
                recorded_kwargs,
                capture_state=snapshot,
                check_state=state_spec.check if state_spec is not None else None,
            )
            if outcome.exception is not None:
                raise outcome.exception
            return outcome.value

        wrapper.identity = identity
        setattr(wrapper, MARKER, RecorderInfo(identity, kind, state_spec))
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
    invocation and must be recorded the same way, which they only are once bound against
    the signature. Returns None if the arguments don't fit the signature at all.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    return bound
