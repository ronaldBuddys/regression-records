"""The public decorators: :func:`record_func` and :func:`record_method`."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from ._engine import RecorderConfig, record_call
from ._env import env_names
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
    env: Sequence[str] = (),
    secret_env: Sequence[str] = (),
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
    env : Sequence[str], default ()
        Environment variables the function reads during a call. Each record holds
        their values (None for one unset), and replay sets them again around the call,
        so the call is replayed under the setting it was recorded under. They are part
        of the input hash: the same arguments under another value are another case.
        Values are stored in plain text, so never list a secret here. A value read at
        import, into a module constant, is not covered: replay sets the variable too
        late for it.
    secret_env : Sequence[str], default ()
        Environment variables holding secrets, such as an API key. Each record holds a
        one-way fingerprint of their values, not the values; replay cannot set them,
        and reports in :attr:`ReplayCase.secret_env_changed
        <regression_records.ReplayCase.secret_env_changed>` those whose current value
        differs. Suited to long random keys: a short or guessable value can be found
        from its fingerprint by trying candidates.

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
        ``staticmethod`` / ``classmethod`` object; or to a function replay could not
        import again by name - a lambda, a function defined inside another, a
        ``functools.partial``, or one defined in the script being run (``__main__``).
        When ``env`` or ``secret_env`` is a single string rather than a sequence of
        names.
    ValueError
        When a name in ``env`` or ``secret_env`` is not a variable name, is listed in
        both, or is ``REGRESSION_RECORDS``.
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
        env=env,
        secret_env=secret_env,
    )


def record_method(
    store: RecordStore | str | Path,
    *,
    state_attrs: Sequence[str] | None = None,
    exclude_attrs: Sequence[str] = (),
    factory: Callable[[], object] | None = None,
    record_exceptions: bool = True,
    tags: Mapping[str, str] | None = None,
    serializer_registry: SerializerRegistry | None = None,
    enabled: bool = True,
    env: Sequence[str] = (),
    secret_env: Sequence[str] = (),
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
    factory : Callable[[], object], optional
        Builds a receiver for replay, which then sets the recorded attributes on it.
        Required with ``state_attrs`` or ``exclude_attrs``: a record holding only part
        of an object's state cannot rebuild the object on its own. Without either,
        replay rebuilds the receiver from its full recorded state, bypassing
        ``__init__``.
    record_exceptions, tags, serializer_registry, enabled, env, secret_env
        Exactly as in :func:`record_func`. Replay sets ``env`` before building the
        receiver, so a ``factory`` sees the recorded values too.

    Returns
    -------
    Callable
        A decorator, whose result carries ``identity`` as for :func:`record_func`.

    Raises
    ------
    TypeError
        When ``state_attrs`` or ``exclude_attrs`` is given without ``factory``, or
        ``factory`` is not callable. When applied to a generator or async function, to
        a method replay could not import (as for :func:`record_func`), or above
        ``@classmethod`` / ``@staticmethod`` rather than beneath it. A staticmethod has
        no receiver: use :func:`record_func`. When ``env`` or ``secret_env`` is a
        single string.
    ValueError
        As for :func:`record_func`.
    SerializationError
        On a recorded call whose receiver exposes no dict-shaped state, whose state
        cannot be deep-copied, or for which ``state_attrs`` names nothing.
    """
    partial_state = state_attrs is not None or bool(exclude_attrs)
    if partial_state and factory is None:
        raise TypeError(
            "record_method with state_attrs or exclude_attrs records only part of the "
            "object's state, which replay cannot rebuild the object from; pass "
            "factory=, a zero-argument callable returning an instance to set the "
            "recorded attributes on"
        )
    if factory is not None and not callable(factory):
        raise TypeError(f"factory must be callable, not {type(factory).__name__}")
    return _build_decorator(
        kind="method",
        store=store,
        enabled=enabled,
        record_exceptions=record_exceptions,
        serializer_registry=serializer_registry,
        tags=tags,
        state_spec=StateSpec.build(state_attrs, exclude_attrs),
        factory=factory,
        env=env,
        secret_env=secret_env,
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
    factory: Callable[[], object] | None = None,
    env: Sequence[str] = (),
    secret_env: Sequence[str] = (),
) -> Callable[[F], F]:
    env_vars = env_names(env, "env")
    secret_vars = env_names(secret_env, "secret_env")
    if both := sorted(set(env_vars) & set(secret_vars)):
        raise ValueError(
            f"{', '.join(both)} listed in both env and secret_env; a variable is "
            "either stored or fingerprinted"
        )
    config = RecorderConfig(
        store=coerce_store(store, serializer_registry),
        record_exceptions=record_exceptions,
        tags=dict(tags or {}),
        env=env_vars,
        secret_env=secret_vars,
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
        setattr(wrapper, MARKER, RecorderInfo(identity, kind, state_spec, factory))
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
    _require_importable(target, decorator)


def _require_importable(target: Any, decorator: str) -> None:
    """Reject a callable that replay could not import again by its name.

    Replay finds a recorded function by ``module`` and ``qualname``, so a function
    without a module-level name would be recorded into a store no replay can check.
    Refused when decorated, rather than discovered at the first replay.

    Raises
    ------
    TypeError
        For a ``functools.partial`` or another callable with no ``__qualname__``; for a
        lambda, or a function or class defined inside a function (``<lambda>`` or
        ``<locals>`` in its qualname); and for one defined in the script being run
        (module ``__main__``), which is a different module when replay runs.
    """
    qualname = getattr(target, "__qualname__", None)
    if qualname is None:
        raise TypeError(
            f"{decorator} cannot record {target!r}: it has no __qualname__ (a "
            "functools.partial, for one), so replay could not import it again. "
            "Decorate the underlying function."
        )
    if "<" in qualname:
        raise TypeError(
            f"{decorator} cannot record '{qualname}': a lambda, or a function or class "
            "defined inside a function, has no name replay can import it by. Define it "
            "at module level."
        )
    if target.__module__ == "__main__":
        raise TypeError(
            f"{decorator} cannot record '{qualname}': it is defined in the script being "
            "run (module __main__), which replay cannot import. Move it into an "
            "importable module and import it from the script."
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
