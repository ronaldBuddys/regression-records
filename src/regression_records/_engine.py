"""The shared record/validate engine behind both decorators.

``record_func`` and ``record_method`` differ only in how they derive an identity, split
their arguments, and (for methods) capture object state. Everything after that - hashing
inputs, choosing between recording and validating, comparing, honoring the overwrite
policy, and re-raising - is the same, and lives here.
"""

from __future__ import annotations

import traceback
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .comparison import ComparisonResult, compare, compare_exceptions
from .comparison.base import ComparatorRegistry, Diff
from .comparison.paths import OUTPUT_ROOT, STATE_AFTER_ROOT, STATE_BEFORE_ROOT
from .comparison.tolerance import ToleranceConfig
from .config import OnMismatch, OnNewCase, OverwritePolicy, RunMode
from .exceptions import CodeDriftWarning, RegressionMismatchError
from .identity import FunctionIdentity
from .matching import Action, decide_action, find_match
from .records import ExceptionRecord, Invocation, RegressionRecord, SerializedValue
from .serialization import to_json_tree
from .serialization.base import SerializerRegistry
from .store import RecordStore


def coerce_store(
    store: RecordStore | str | Path, registry: SerializerRegistry | None = None
) -> RecordStore:
    """Accept either a shared :class:`RecordStore` or a path to wrap in a new one."""
    if isinstance(store, RecordStore):
        return store
    return RecordStore(store, serializer_registry=registry)


@dataclass(frozen=True)
class RecorderConfig:
    """Everything the engine needs that isn't derived from the call itself."""

    store: RecordStore
    mode: RunMode = RunMode.BOTH
    tolerance: ToleranceConfig | None = None
    on_new_case: OnNewCase = OnNewCase.RECORD
    overwrite: OverwritePolicy = OverwritePolicy.NEVER
    on_mismatch: OnMismatch = OnMismatch.RAISE
    record_exceptions: bool = True
    comparator_registry: ComparatorRegistry | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validated by the config that owns the invariant rather than by each caller,
        # so a second entry point into the engine can't quietly skip the check and
        # degrade to "return" on a typo.
        if self.on_mismatch not in tuple(OnMismatch):
            raise ValueError(
                f"on_mismatch must be one of {[m.value for m in OnMismatch]}, "
                f"got {self.on_mismatch!r}"
            )


@dataclass
class Outcome:
    """What one recorded invocation produced.

    ``exception`` is the *real* exception the wrapped callable raised, kept so the
    caller can re-raise it unchanged - recording never swallows or substitutes a
    function's behavior.
    """

    value: Any = None
    exception: BaseException | None = None
    comparison: ComparisonResult | None = None


def execute(
    config: RecorderConfig,
    identity: FunctionIdentity,
    invoke: Callable[[], Any],
    args_value: Any,
    kwargs_value: Mapping[str, Any],
    *,
    capture_state: Callable[[], Any] | None = None,
) -> Outcome:
    """Run one invocation through the record/validate pipeline.

    ``invoke`` calls the underlying callable with its original arguments untouched;
    ``args_value``/``kwargs_value`` are the normalized forms used only for hashing and
    for what gets written to the record. ``capture_state``, when given, is called once
    before and once after ``invoke`` (used by ``record_method``).
    """
    ser_ctx = config.store.serialization_context(identity, deferred=True)
    committed = False
    try:
        args_tree = to_json_tree(list(args_value), ser_ctx)
        kwargs_tree = to_json_tree(dict(kwargs_value), ser_ctx)

        match = find_match(config.store, identity, args_tree, kwargs_tree)
        action = decide_action(match, mode=config.mode, on_new_case=config.on_new_case)
        if action is Action.SKIP:
            return Outcome(value=invoke())

        state_before = capture_state() if capture_state is not None else None
        outcome = _invoke(invoke)
        state_after = _capture_after(capture_state, outcome)

        if outcome.exception is not None and not config.record_exceptions:
            return outcome

        if action is Action.VALIDATE:
            committed = _validate(
                config,
                identity,
                match.record,
                outcome,
                state_before,
                state_after,
                ser_ctx,
            )
        else:
            committed = _record(
                config,
                identity,
                match,
                outcome,
                state_before,
                state_after,
                ser_ctx,
                args_tree,
                kwargs_tree,
            )
        return outcome
    finally:
        if committed:
            ser_ctx.flush()
        else:
            # Nothing referenced these side files, so writing them would only leave
            # unreferenced data behind in the store.
            ser_ctx.discard()


def _invoke(invoke: Callable[[], Any]) -> Outcome:
    """Call the wrapped callable, capturing an ordinary exception rather than raising.

    A ``BaseException`` that isn't an ``Exception`` (``KeyboardInterrupt``,
    ``SystemExit``) deliberately propagates untouched - it signals that the process is
    going away, not that the function under test misbehaved.
    """
    try:
        return Outcome(value=invoke())
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
        # raised, which is the more useful signal.
        return None


def _exception_record(exc: BaseException) -> ExceptionRecord:
    return ExceptionRecord(
        exc_type=type(exc).__name__,
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
    state_before: Any,
    state_after: Any,
    ser_ctx: Any,
    *,
    record_id: str | None = None,
) -> RegressionRecord:
    has_state = state_before is not None or state_after is not None
    return RegressionRecord.new(
        identity=identity,
        input_hash=input_hash,
        invocation=Invocation(
            args=SerializedValue(args_tree), kwargs=SerializedValue(kwargs_tree)
        ),
        output=(
            SerializedValue(to_json_tree(outcome.value, ser_ctx))
            if outcome.exception is None
            else None
        ),
        exception=_exception_record(outcome.exception)
        if outcome.exception is not None
        else None,
        state_before=SerializedValue(to_json_tree(state_before, ser_ctx))
        if has_state
        else None,
        state_after=SerializedValue(to_json_tree(state_after, ser_ctx))
        if has_state
        else None,
        tags=dict(config.tags),
        record_id=record_id,
    )


def _record(
    config: RecorderConfig,
    identity: FunctionIdentity,
    match: Any,
    outcome: Outcome,
    state_before: Any,
    state_after: Any,
    ser_ctx: Any,
    args_tree: Any,
    kwargs_tree: Any,
) -> bool:
    """Write a record for this invocation. Returns whether anything was committed."""
    if match.record is not None and config.overwrite is OverwritePolicy.NEVER:
        # RECORD_ONLY revisiting an already-recorded case: writing again would either
        # duplicate the case under a second record id or rewrite history, and NEVER
        # means history is immutable.
        return False

    record = _build_record(
        config,
        identity,
        match.input_hash,
        args_tree,
        kwargs_tree,
        outcome,
        state_before,
        state_after,
        ser_ctx,
        record_id=match.record.record_id if match.record is not None else None,
    )
    config.store.save(record, overwrite=match.record is not None)
    return True


def _validate(
    config: RecorderConfig,
    identity: FunctionIdentity,
    recorded: RegressionRecord,
    outcome: Outcome,
    state_before: Any,
    state_after: Any,
    ser_ctx: Any,
) -> bool:
    """Compare this invocation against ``recorded``. Returns whether anything was written."""
    _warn_on_code_drift(identity, recorded)
    result = compare_invocation(
        config, identity, recorded, outcome, state_before, state_after
    )
    outcome.comparison = result

    accept = config.overwrite is OverwritePolicy.ALWAYS or (
        config.overwrite is OverwritePolicy.IF_MISMATCH and not result.passed
    )
    if accept:
        # Updating the record *is* accepting the new behavior as correct, so the
        # mismatch is resolved rather than reported - that is the whole point of
        # re-running with IF_MISMATCH after an intentional change. The diffs remain on
        # the wrapper's ``last_result`` for inspection.
        updated = _build_record(
            config,
            identity,
            recorded.input_hash,
            recorded.invocation.args.data,
            recorded.invocation.kwargs.data,
            outcome,
            state_before,
            state_after,
            ser_ctx,
            record_id=recorded.record_id,
        )
        config.store.save(updated, overwrite=True)
        return True

    # When the call itself raised, that exception is the more direct signal and is
    # always what reaches the caller; the mismatch stays on ``last_result``.
    if (
        not result.passed
        and config.on_mismatch == OnMismatch.RAISE
        and outcome.exception is None
    ):
        raise RegressionMismatchError(result)
    return False


def _warn_on_code_drift(identity: FunctionIdentity, recorded: RegressionRecord) -> None:
    """Note when the code under test has changed since the record was captured.

    ``code_hash`` is never part of the matching key - a refactor must not orphan a
    function's history - so this is purely informational, and tells you whether a pass
    means "behavior held across a change" or just "nothing changed".
    """
    if (
        identity.code_hash == "unknown"
        or identity.code_hash == recorded.identity.code_hash
    ):
        return
    warnings.warn(
        f"'{identity.qualname}' has changed since this record was captured "
        f"(code_hash {recorded.identity.code_hash} -> {identity.code_hash}); the "
        "comparison below is still meaningful, but the recorded baseline predates the "
        "current implementation",
        CodeDriftWarning,
        # _warn_on_code_drift -> _validate -> execute -> wrapper -> the caller, who is
        # the one who wants to see which of their call sites this concerns.
        stacklevel=5,
    )


def compare_invocation(
    config: RecorderConfig,
    identity: FunctionIdentity,
    recorded: RegressionRecord,
    outcome: Outcome,
    state_before: Any = None,
    state_after: Any = None,
) -> ComparisonResult:
    """Compare a fresh invocation's result (and state) against a historical record."""
    deser_ctx = config.store.deserialization_context(identity)
    diffs: list[Diff] = []

    diffs.extend(_compare_result(config, recorded, outcome, deser_ctx))
    for recorded_state, fresh_state, root in (
        (recorded.state_before, state_before, STATE_BEFORE_ROOT),
        (recorded.state_after, state_after, STATE_AFTER_ROOT),
    ):
        if recorded_state is None:
            continue
        diffs.extend(
            compare(
                recorded_state.load(deser_ctx),
                fresh_state,
                config.tolerance,
                registry=config.comparator_registry,
                root=root,
            ).diffs
        )
    return ComparisonResult.from_diffs(diffs)


def _compare_result(
    config: RecorderConfig, recorded: RegressionRecord, outcome: Outcome, deser_ctx: Any
) -> list[Diff]:
    """Compare the return value, or the exception raised in its place."""
    if recorded.succeeded and outcome.exception is None:
        return compare(
            recorded.load_output(deser_ctx),
            outcome.value,
            config.tolerance,
            registry=config.comparator_registry,
            root=OUTPUT_ROOT,
        ).diffs

    # At least one side raised, so there is no output to compare - the exception (or its
    # absence) is the whole result.
    return compare_exceptions(
        recorded.exception.exc_type if not recorded.succeeded else None,
        outcome.exception,
        root=OUTPUT_ROOT,
    ).diffs
