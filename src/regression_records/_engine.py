"""The recording engine behind both decorators.

``record_func`` and ``record_method`` differ only in how they derive an identity, split
their arguments, and (for methods) capture object state. Everything after that -
serializing, invoking, building the record and writing it - is the same, and lives
here. Nothing here compares: checking a record against current behavior is replay's job.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .comparison import exception_type_name
from .exceptions import RegressionRecordsError, SerializationError
from .identity import FunctionIdentity
from .records import (
    ExceptionRecord,
    Invocation,
    RegressionRecord,
    SerializedValue,
    compute_input_hash,
    content_record_id,
)
from .serialization import SerializationContext, to_json_tree
from .store import RecordStore


@dataclass(frozen=True)
class RecorderConfig:
    """Everything the engine needs that isn't derived from the call itself."""

    store: RecordStore
    record_exceptions: bool = True
    tags: Mapping[str, str] = field(default_factory=dict)


class _NotCaptured:
    """The after-state of a call whose snapshot failed; see :func:`_capture_after`."""

    def __repr__(self) -> str:
        return "<state not captured>"


#: Distinct from ``None``, which is what "this call records no state" looks like.
NOT_CAPTURED = _NotCaptured()


@dataclass
class Outcome:
    """What one invocation produced.

    ``exception`` is the *real* exception the wrapped callable raised, kept so the
    caller can re-raise it unchanged - recording never swallows or substitutes a
    function's behavior.
    """

    value: Any = None
    exception: BaseException | None = None


#: The calls this process has already recorded, so that a call repeated in a loop is
#: serialized and written once. Keyed on what the call was given - store, identity and
#: code, arguments and, for a method, the receiver's state - not on what it returned.
_recorded: set[tuple[str, ...]] = set()
_recorded_lock = threading.Lock()


def forget_recorded_calls() -> None:
    """Forget which calls this process has recorded, so the next of each records again.

    For tests, which need a fresh process's behavior without starting one.
    """
    with _recorded_lock:
        _recorded.clear()


def record_call(
    config: RecorderConfig,
    identity: FunctionIdentity,
    invoke: Callable[[], Any],
    args_value: Any,
    kwargs_value: Mapping[str, Any],
    *,
    capture_state: Callable[[], Any] | None = None,
    check_state: Callable[[Any, Any], None] | None = None,
) -> Outcome:
    """Run one invocation and record it.

    ``invoke`` calls the underlying callable with its original arguments untouched;
    ``args_value``/``kwargs_value`` are the normalized forms that get recorded.
    ``capture_state``, when given, is called once before and once after ``invoke``
    (used by ``record_method``), and ``check_state`` is then handed both snapshots to
    reject a selection that captured nothing.

    The record's id is derived from its content, and the store never replaces a record:
    recording behavior identical to an earlier recording writes nothing, and a changed
    behavior adds a record beside the old one.
    """
    ser_ctx = config.store.serialization_context(identity, deferred=True)
    try:
        args_tree = to_json_tree(list(args_value), ser_ctx)
        kwargs_tree = to_json_tree(dict(kwargs_value), ser_ctx)
        input_hash = compute_input_hash(args_tree, kwargs_tree)

        state_before = capture_state() if capture_state is not None else None
        before_tree = (
            to_json_tree(state_before, ser_ctx) if capture_state is not None else None
        )
        call_key = _call_key(config.store, identity, input_hash, before_tree)

        outcome = _invoke(invoke)
        state_after = _capture_after(capture_state, outcome)
        if check_state is not None and state_after is not NOT_CAPTURED:
            check_state(state_before, state_after)

        if outcome.exception is not None and not config.record_exceptions:
            return outcome
        with _recorded_lock:
            if call_key in _recorded:
                return outcome

        record = _build_record(
            config,
            identity,
            input_hash,
            args_tree,
            kwargs_tree,
            outcome,
            before_tree,
            state_after,
            ser_ctx,
        )
        _commit(config, record, ser_ctx)
        with _recorded_lock:
            _recorded.add(call_key)
        return outcome
    finally:
        # A committed record flushed its side files before it was saved, so anything
        # still buffered is referenced by nothing, and writing it would only leave
        # orphaned data behind in the store.
        ser_ctx.discard()


def _call_key(
    store: RecordStore, identity: FunctionIdentity, input_hash: str, before_tree: Any
) -> tuple[str, ...]:
    # Two instances in different states making the same call are different cases, so
    # a method's key includes its state. The code hash is in it so that redefining the
    # function - an intended change, made in a notebook or REPL - records again in the
    # same process rather than being taken for a repeat.
    state_hash = (
        compute_input_hash([before_tree], {}) if before_tree is not None else ""
    )
    return (
        str(store.root.resolve()),
        identity.module,
        identity.qualname,
        identity.code_hash,
        input_hash,
        state_hash,
    )


def _invoke(invoke: Callable[[], Any]) -> Outcome:
    """Call the wrapped callable, capturing an ordinary exception rather than raising.

    A ``BaseException`` that isn't an ``Exception`` (``KeyboardInterrupt``,
    ``SystemExit``) deliberately propagates untouched - it signals that the process is
    going away, not that the function under test misbehaved.

    So does this library's own error. It means a decorated call *inside* the callable
    could not be recorded, and capturing it would record "raises SerializationError"
    as this callable's behavior.
    """
    try:
        return Outcome(value=invoke())
    except RegressionRecordsError:
        raise
    except Exception as exc:  # noqa: BLE001 - deliberately broad; re-raised by the caller
        return Outcome(exception=exc)


def _capture_after(capture_state: Callable[[], Any] | None, outcome: Outcome) -> Any:
    """Snapshot state after the call, including when the call raised.

    A method that failed part-way may still have mutated its object.
    """
    if capture_state is None:
        return None
    try:
        return capture_state()
    except Exception:
        if outcome.exception is None:
            raise
        # A failure to snapshot state must not mask the exception the call itself
        # raised, which is the more useful signal. Not None: that would be recorded as
        # a state of null, which no later run's real state could ever match.
        return NOT_CAPTURED


def _exception_record(exc: BaseException) -> ExceptionRecord:
    return ExceptionRecord(
        exc_type=exception_type_name(exc),
        message=str(exc),
        traceback_text="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    )


def _build_record(
    config: RecorderConfig,
    identity: FunctionIdentity,
    input_hash: str,
    args_tree: Any,
    kwargs_tree: Any,
    outcome: Outcome,
    before_tree: Any,
    state_after: Any,
    ser_ctx: SerializationContext,
) -> RegressionRecord:
    output_tree = (
        to_json_tree(outcome.value, ser_ctx) if outcome.exception is None else None
    )
    exception = (
        _exception_record(outcome.exception) if outcome.exception is not None else None
    )
    has_state_after = before_tree is not None and state_after is not NOT_CAPTURED
    after_tree = to_json_tree(state_after, ser_ctx) if has_state_after else None
    return RegressionRecord.new(
        identity=identity,
        input_hash=input_hash,
        invocation=Invocation(
            args=SerializedValue(args_tree), kwargs=SerializedValue(kwargs_tree)
        ),
        output=SerializedValue(output_tree) if exception is None else None,
        exception=exception,
        state_before=SerializedValue(before_tree) if before_tree is not None else None,
        state_after=SerializedValue(after_tree) if has_state_after else None,
        tags=dict(config.tags),
        record_id=content_record_id(
            module=identity.module,
            qualname=identity.qualname,
            input_hash=input_hash,
            output_tree=output_tree,
            exc_type=exception.exc_type if exception is not None else None,
            state_before_tree=before_tree,
            state_after_tree=after_tree,
        ),
    )


def _commit(
    config: RecorderConfig, record: RegressionRecord, ser_ctx: SerializationContext
) -> None:
    """Write ``record``'s side files, then the record that references them.

    In that order, so an interruption between the two leaves unreferenced data files
    (harmless, reused by the next attempt since they are content-addressed, and
    collected by ``RecordStore.prune``) rather than a record pointing at files that do
    not exist.

    A record file that already exists holds this same behavior, since the id is
    derived from it: another run, or a parallel worker, recorded it first.

    Raises
    ------
    SerializationError
        If the existing file is a different record whose id shares this one's
        filename prefix, which is astronomically unlikely but would otherwise lose a
        recording silently.
    """
    ser_ctx.flush()
    try:
        config.store.save(record)
    except FileExistsError as e:
        existing = config.store.load_record(e.filename)
        if existing.record_id != record.record_id:
            raise SerializationError(
                f"cannot record {record.identity.module}:{record.identity.qualname}: "
                f"its filename is taken by a different record, {e.filename}"
            ) from e
