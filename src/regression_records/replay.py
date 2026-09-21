"""Replay recorded calls against the current code, and report what changed.

Recording captures what a function did: its arguments, and what it returned or raised.
:func:`replay` checks, possibly much later and in a different environment, that it still
does. For each record in a store it imports the function by its recorded identity,
calls the original (undecorated) function on the recorded arguments, and reconciles the
result against the recorded one. A method is called on a receiver rebuilt from its
recorded state, and its state after the call is reconciled too. Every record is checked
on its own; nothing is looked up by input hash, so two records of the same call that
disagree simply give one pass and one failure.

Replay only observes. It calls the function beneath the decorator, and while it runs no
decorated call anywhere in the process records, whatever ``REGRESSION_RECORDS`` says.
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._introspect import capture_state
from ._reconcile import reconcile
from ._switch import MARKER, RecorderInfo, suppressed
from .comparison import ComparisonResult
from .comparison.base import ComparatorRegistry
from .comparison.paths import STATE_AFTER_ROOT
from .comparison.tolerance import ToleranceConfig
from .exceptions import (
    CorruptRecordError,
    DeserializationError,
    IdentityResolutionError,
)
from .identity import FunctionIdentity
from .records import RegressionRecord
from .store import RecordStore, coerce_store

#: A record filter: a glob on ``"module:qualname"``, or a predicate on the record.
Selector = str | Callable[[RegressionRecord], bool]


class ReplayStatus(StrEnum):
    """How one record fared on replay."""

    #: The call behaved as recorded, within tolerance.
    PASSED = "passed"
    #: The result, the state, or the exception raised differs from the record. This
    #: includes a call that returned where the record says it raised.
    DIFFERS = "differs"
    #: The record says the call returned, but on replay it raised. Kept apart from
    #: DIFFERS because it often means the call could not run faithfully outside its
    #: original context (a clock, a database, a global) or its signature changed.
    RAISED = "raised"
    #: The recorded module or function no longer exists under its recorded name.
    NOT_IMPORTABLE = "not_importable"
    #: The recorded name resolves, but not to a recorded function that replay can call.
    NOT_REPLAYABLE = "not_replayable"
    #: The record file, or a value it holds, cannot be read.
    NOT_LOADABLE = "not_loadable"


@dataclass(frozen=True)
class ReplayCase:
    """The replay of one record.

    Attributes
    ----------
    record_path : str
        The record file, relative to the store root, with ``/`` separators.
    status : ReplayStatus
        The outcome.
    identity : FunctionIdentity or None
        The recorded function; None when the record could not be read.
    record_id, input_hash, created_at : str or None
        Copied from the record, so that records of the same call can be grouped
        without replay deciding anything about them; None when it could not be read.
    comparison : ComparisonResult or None, default None
        The reconciliation, for ``PASSED``, ``DIFFERS`` and ``RAISED``. Its diffs are
        capped at ``max_diffs``; see ``diffs_truncated``.
    error : str or None, default None
        Why the case did not pass, when that is an error rather than a diff: the
        exception raised on replay, or why the record could not be loaded, imported or
        called.
    code_drift : bool, default False
        Whether the function's source has changed since the record was made. Replay
        runs current code on old inputs, so drift is expected; it says whether a pass
        confirms unchanged behavior across a real change, or merely that nothing
        changed.
    recorded_code_hash, current_code_hash : str or None, default None
        The two sides of ``code_drift``.
    diffs_truncated : int, default 0
        How many further diffs were found and dropped from ``comparison``.
    tags : Mapping[str, str], default {}
        The record's tags.
    """

    record_path: str
    status: ReplayStatus
    identity: FunctionIdentity | None = None
    record_id: str | None = None
    input_hash: str | None = None
    created_at: str | None = None
    comparison: ComparisonResult | None = None
    error: str | None = None
    code_drift: bool = False
    recorded_code_hash: str | None = None
    current_code_hash: str | None = None
    diffs_truncated: int = 0
    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return whether the status is ``PASSED``."""
        return self.status is ReplayStatus.PASSED

    @property
    def name(self) -> str:
        """Return ``"module:qualname"``, or the record path when it could not be read."""
        if self.identity is None:
            return self.record_path
        return f"{self.identity.module}:{self.identity.qualname}"

    def describe(self, max_diffs: int = 20) -> str:
        """Return a readable account of this case.

        Parameters
        ----------
        max_diffs : int, default 20
            How many diffs to list.

        Returns
        -------
        str
            A header line naming the status, the function and the record, then the
            error and the diffs, if any.
        """
        record = f" record {self.record_id[:8]}" if self.record_id else ""
        lines = [f"{self.status.value}: {self.name}{record} ({self.record_path})"]
        if self.code_drift:
            lines.append(
                f"  code changed since recording ({self.recorded_code_hash} -> "
                f"{self.current_code_hash})"
            )
        if self.error:
            lines.append(f"  {self.error}")
        if self.comparison is not None and not self.comparison.passed:
            summary = self.comparison.summary(max_diffs)
            lines.extend(f"  {line}" for line in summary.splitlines())
            if self.diffs_truncated:
                lines.append(f"  ({self.diffs_truncated} further diff(s) not kept)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict.

        Returns
        -------
        dict
            Every attribute, with ``status`` as its string value, ``identity`` as a
            dict, and ``comparison`` as ``{"passed": ..., "diffs": [...]}``.
        """
        return {
            "record_path": self.record_path,
            "status": self.status.value,
            "identity": asdict(self.identity) if self.identity is not None else None,
            "record_id": self.record_id,
            "input_hash": self.input_hash,
            "created_at": self.created_at,
            "comparison": (
                {
                    "passed": self.comparison.passed,
                    "diffs": [asdict(d) for d in self.comparison.diffs],
                }
                if self.comparison is not None
                else None
            ),
            "error": self.error,
            "code_drift": self.code_drift,
            "recorded_code_hash": self.recorded_code_hash,
            "current_code_hash": self.current_code_hash,
            "diffs_truncated": self.diffs_truncated,
            "tags": dict(self.tags),
        }


@dataclass(frozen=True)
class ReplayReport:
    """The replay of every selected record in a store.

    Attributes
    ----------
    store_root : Path
        The store replayed.
    cases : tuple[ReplayCase, ...]
        One per record, in record-path order.
    """

    store_root: Path
    cases: tuple[ReplayCase, ...]

    @property
    def passed(self) -> bool:
        """Return whether every case passed; True for a store with no records."""
        return all(case.passed for case in self.cases)

    @property
    def failures(self) -> tuple[ReplayCase, ...]:
        """Return the cases that did not pass."""
        return tuple(case for case in self.cases if not case.passed)

    def counts(self) -> dict[str, int]:
        """Return the number of cases per status, every status included.

        Returns
        -------
        dict[str, int]
            Keyed by each :class:`ReplayStatus` value, in declaration order.
        """
        tally = Counter(case.status for case in self.cases)
        return {status.value: tally[status] for status in ReplayStatus}

    def summary(self, max_cases: int = 20, max_diffs: int = 5) -> str:
        """Return a readable report: the counts, then each failure.

        Parameters
        ----------
        max_cases : int, default 20
            How many failures to describe before summarizing the rest.
        max_diffs : int, default 5
            How many diffs to list per failure.

        Returns
        -------
        str
            ``"replayed N record(s) from <root>: ..."`` with the non-zero counts, then
            one :meth:`ReplayCase.describe` block per listed failure.
        """
        counts = ", ".join(f"{n} {s}" for s, n in self.counts().items() if n)
        lines = [
            f"replayed {len(self.cases)} record(s) from {self.store_root}: "
            f"{counts or 'nothing to replay'}"
        ]
        failures = self.failures
        lines.extend(case.describe(max_diffs) for case in failures[:max_cases])
        if len(failures) > max_cases:
            lines.append(f"... and {len(failures) - max_cases} more failure(s)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict, for a later tool to track results over time.

        Returns
        -------
        dict
            ``{"store_root": str, "counts": ..., "cases": [ReplayCase.to_dict(), ...]}``.
        """
        return {
            "store_root": str(self.store_root),
            "counts": self.counts(),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class _Options:
    """The settings shared by every case of one replay."""

    tolerance: ToleranceConfig | None = None
    tolerance_overrides: Mapping[str, ToleranceConfig] = field(default_factory=dict)
    comparator_registry: ComparatorRegistry | None = None
    max_diffs: int = 100

    def tolerance_for(self, name: str) -> ToleranceConfig | None:
        """Return the first override whose pattern matches ``name``, else the default."""
        for pattern, tolerance in self.tolerance_overrides.items():
            if fnmatch.fnmatchcase(name, pattern):
                return tolerance
        return self.tolerance


def replay(
    store: RecordStore | str | Path,
    *,
    tolerance: ToleranceConfig | None = None,
    tolerance_overrides: Mapping[str, ToleranceConfig] | None = None,
    select: Selector | None = None,
    comparator_registry: ComparatorRegistry | None = None,
    max_diffs: int = 100,
) -> ReplayReport:
    """Rerun every recorded call in ``store`` and reconcile it with its record.

    Each record is replayed independently: its function is imported by the recorded
    identity, the original function beneath the decorator is called on the recorded
    arguments, and what it returns or raises is compared with what was recorded.
    Nothing is written to the store, and no decorated call records while replay runs.

    A method record is called on a receiver rebuilt in its recorded state before the
    call - by the method's ``factory=`` when it has one, else from the recorded state
    alone, bypassing ``__init__`` - and its state after the call is compared as well
    (``$.state_after``). The current decorator's ``state_attrs``, ``exclude_attrs``
    and ``factory`` decide how; ``state_before`` is not compared.

    Parameters
    ----------
    store : RecordStore or str or Path
        The store to replay. Pass a :class:`RecordStore` to use a custom serializer
        registry.
    tolerance : ToleranceConfig, optional
        Numeric tolerances for every case not matched by ``tolerance_overrides``; the
        defaults when omitted.
    tolerance_overrides : Mapping[str, ToleranceConfig], optional
        Tolerances for particular functions, keyed by a glob on ``"module:qualname"``
        (``"pricing.*:price_*"``). The first matching pattern wins.
    select : str or Callable[[RegressionRecord], bool], optional
        Replay only some records: a glob on ``"module:qualname"``, or a predicate on the
        record. A record that cannot be read is matched by a glob against the name its
        path implies, and is always included under a predicate, so that a corrupt
        record is never silently skipped.
    comparator_registry : ComparatorRegistry, optional
        The comparators to use; the default registry when omitted.
    max_diffs : int, default 100
        The most diffs kept per case; the rest are counted in
        :attr:`ReplayCase.diffs_truncated`.

    Returns
    -------
    ReplayReport
        One :class:`ReplayCase` per selected record, in record-path order.

    Raises
    ------
    FileNotFoundError
        If ``store`` does not exist: a mistyped path would otherwise replay nothing
        and pass.

    Notes
    -----
    A call is replayed from its recorded inputs alone. A function that reads a clock,
    a database, an environment variable or other global state sees today's value, not
    the recorded one, and its side effects happen again.

    Examples
    --------
    >>> from regression_records import ToleranceConfig, replay
    >>> report = replay("tests/records", tolerance=ToleranceConfig(rtol=1e-9))
    >>> assert report.passed, report.summary()
    """
    record_store, options, paths = _plan_replay(
        store,
        tolerance=tolerance,
        tolerance_overrides=tolerance_overrides,
        select=select,
        comparator_registry=comparator_registry,
        max_diffs=max_diffs,
    )
    cases = tuple(_replay_record(record_store, path, options) for path in paths)
    return ReplayReport(store_root=record_store.root, cases=cases)


def _plan_replay(
    store: RecordStore | str | Path,
    *,
    tolerance: ToleranceConfig | None = None,
    tolerance_overrides: Mapping[str, ToleranceConfig] | None = None,
    select: Selector | None = None,
    comparator_registry: ComparatorRegistry | None = None,
    max_diffs: int = 100,
) -> tuple[RecordStore, _Options, list[Path]]:
    """Resolve :func:`replay`'s arguments into the store, settings and record paths.

    Shared with :func:`~regression_records.testing.replay_tests`, which collects the
    paths up front and replays each one as its own test.

    Raises
    ------
    FileNotFoundError
        If ``store`` does not exist.
    """
    record_store = coerce_store(store)
    if not record_store.root.is_dir():
        raise FileNotFoundError(f"no record store at {record_store.root}")
    options = _Options(
        tolerance=tolerance,
        tolerance_overrides=dict(tolerance_overrides or {}),
        comparator_registry=comparator_registry,
        max_diffs=max_diffs,
    )
    paths = list(_selected(record_store, record_store.iter_record_paths(), select))
    return record_store, options, paths


def _selected(
    store: RecordStore, paths: Iterable[Path], select: Selector | None
) -> Iterable[Path]:
    """Yield the record paths ``select`` keeps."""
    for path in paths:
        if select is None:
            yield path
            continue
        try:
            record = store.load_record(path)
        except CorruptRecordError:
            # Unreadable, so only its location says what it is.
            if callable(select) or fnmatch.fnmatchcase(
                _name_from_path(store, path), select
            ):
                yield path
            continue
        name = f"{record.identity.module}:{record.identity.qualname}"
        if select(record) if callable(select) else fnmatch.fnmatchcase(name, select):
            yield path


def _name_from_path(store: RecordStore, path: Path) -> str:
    """Return the ``"module:qualname"`` a record's location implies.

    ``<root>/<module parts>/<qualname>/records/<file>``: only a fallback for a record
    that cannot be read, since a record is filed by its identity, not defined by where
    it sits.
    """
    parts = path.relative_to(store.root).parts[:-2]
    return f"{'.'.join(parts[:-1])}:{parts[-1]}" if len(parts) > 1 else "/".join(parts)


def _replay_record(store: RecordStore, path: Path, options: _Options) -> ReplayCase:
    """Replay the single record at ``path``, reporting every failure in the case."""
    relative = path.relative_to(store.root).as_posix()
    try:
        record = store.load_record(path)
    except CorruptRecordError as e:
        return ReplayCase(relative, ReplayStatus.NOT_LOADABLE, error=str(e))

    def case(status: ReplayStatus, **fields: Any) -> ReplayCase:
        return ReplayCase(
            record_path=relative,
            status=status,
            identity=record.identity,
            record_id=record.record_id,
            input_hash=record.input_hash,
            created_at=record.created_at,
            recorded_code_hash=record.identity.code_hash,
            tags=dict(record.tags),
            **fields,
        )

    # Imported before any value is loaded: importing the user's module is what
    # registers their serializers and makes their classes importable.
    try:
        resolved = record.identity.resolve()
    except IdentityResolutionError as e:
        return case(ReplayStatus.NOT_IMPORTABLE, error=str(e))

    info, reason = _recorder_info(resolved, record.identity)
    if info is None:
        return case(ReplayStatus.NOT_REPLAYABLE, error=reason)
    original = info.original

    current_code_hash = FunctionIdentity.from_function(original).code_hash
    drift = {
        "current_code_hash": current_code_hash,
        "code_drift": "unknown" not in (current_code_hash, record.identity.code_hash)
        and current_code_hash != record.identity.code_hash,
    }

    deser_ctx = store.deserialization_context(record.identity)
    try:
        args = record.load_args(deser_ctx)
        kwargs = record.load_kwargs(deser_ctx)
    except Exception as e:  # noqa: BLE001 - a custom serializer can fail any way
        return case(
            ReplayStatus.NOT_LOADABLE,
            error=f"cannot load the recorded arguments: {type(e).__name__}: {e}",
            **drift,
        )

    if info.recorder.kind == "method":
        try:
            state_before = record.load_state_before(deser_ctx)
            # A factory is user code, which may itself call recorded functions.
            with suppressed():
                receiver = _rebuild_receiver(record, info.recorder, state_before)
        except (DeserializationError, IdentityResolutionError) as e:
            return case(
                ReplayStatus.NOT_LOADABLE,
                error=f"cannot load the recorded state: {type(e).__name__}: {e}",
                **drift,
            )
        except Exception as e:  # noqa: BLE001 - __new__, __setstate__ or a factory
            return case(
                ReplayStatus.NOT_REPLAYABLE,
                error=f"cannot rebuild the receiver: {type(e).__name__}: {e}",
                **drift,
            )
        args = [receiver, *args]

    value, raised = _call(original, args, kwargs)
    states: dict[str, Any] = {}
    if info.recorder.kind == "method" and record.state_after is not None:
        try:
            states[STATE_AFTER_ROOT] = capture_state(args[0], info.recorder.state_spec)
        except Exception as e:  # noqa: BLE001 - reported like any failure to run
            return case(
                ReplayStatus.RAISED,
                error=f"cannot snapshot the receiver after replay: {e}",
                **drift,
            )
    try:
        result = reconcile(
            record,
            value,
            raised,
            deser_ctx,
            tolerance=options.tolerance_for(
                f"{record.identity.module}:{record.identity.qualname}"
            ),
            comparator_registry=options.comparator_registry,
            states=states,
        )
    except (DeserializationError, IdentityResolutionError) as e:
        return case(
            ReplayStatus.NOT_LOADABLE,
            error=f"cannot load the recorded result: {type(e).__name__}: {e}",
            **drift,
        )

    comparison, truncated = _capped(result, options.max_diffs)
    if result.passed:
        return case(ReplayStatus.PASSED, comparison=comparison, **drift)
    if record.succeeded and raised is not None:
        return case(
            ReplayStatus.RAISED,
            comparison=comparison,
            error=f"raised on replay: {type(raised).__name__}: {raised}",
            diffs_truncated=truncated,
            **drift,
        )
    return case(
        ReplayStatus.DIFFERS,
        comparison=comparison,
        diffs_truncated=truncated,
        **drift,
    )


@dataclass(frozen=True)
class _Resolved:
    """A recorder found in a wrapper chain, and the function it wraps."""

    recorder: RecorderInfo
    original: Callable[..., Any]


def _recorder_info(
    resolved: Any, identity: FunctionIdentity
) -> tuple[_Resolved | None, str]:
    """Find the recorder in ``resolved``'s wrapper chain, and the function it wraps.

    Replay calls the function beneath the recording decorator, never whatever the name
    happens to resolve to now: a name that no longer holds a recorded function is
    reported, not called.

    Returns
    -------
    tuple
        ``(found, "")``, or ``(None, reason)``.
    """
    obj, seen = resolved, set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        info = getattr(obj, MARKER, None)
        if isinstance(info, RecorderInfo):
            recorded_as = (info.identity.module, info.identity.qualname)
            if recorded_as != (identity.module, identity.qualname):
                return None, (
                    f"'{identity.module}:{identity.qualname}' now holds the recorder of "
                    f"'{recorded_as[0]}:{recorded_as[1]}'"
                )
            if info.kind != identity.kind:
                return None, (
                    f"'{identity.module}:{identity.qualname}' was recorded as a "
                    f"{identity.kind} but is now recorded as a {info.kind}"
                )
            return _Resolved(info, obj.__wrapped__), ""
        obj = getattr(obj, "__wrapped__", None)
    decorator = "record_method" if identity.kind == "method" else "record_func"
    return None, (
        f"'{identity.module}:{identity.qualname}' is no longer a recorded "
        f"{identity.kind} (it is not decorated with {decorator})"
    )


def _rebuild_receiver(
    record: RegressionRecord,
    recorder: RecorderInfo,
    state_before: dict[str, Any],
) -> Any:
    """Rebuild the object a recorded method was called on, in its recorded state.

    With a ``factory``, the object it builds gets the recorded attributes set on top -
    the record may hold only some of them. Without one, the record holds the object's
    whole state, so it is rebuilt from that alone, bypassing ``__init__`` as unpickling
    does: ``__setstate__`` when the class defines one, else the instance ``__dict__``.

    Raises
    ------
    IdentityResolutionError
        If the method's class can no longer be imported.
    TypeError
        If the factory returns something other than an instance of that class.
    """
    owner, _, _ = record.identity.qualname.rpartition(".")
    cls = replace(record.identity, qualname=owner).resolve()
    if recorder.factory is not None:
        receiver = recorder.factory()
        if not isinstance(receiver, cls):
            raise TypeError(
                f"factory returned {type(receiver).__qualname__}, not an instance of "
                f"{cls.__qualname__}"
            )
        # Only what the current decorator selects: a record made before an attribute
        # was excluded still holds it, and must not overwrite what the factory built
        # (a live connection, say) with a recorded copy.
        for name, value in state_before.items():
            if recorder.state_spec is None or recorder.state_spec.selects(name):
                setattr(receiver, name, value)
        return receiver
    receiver = cls.__new__(cls)
    setstate = getattr(receiver, "__setstate__", None)
    if setstate is not None:
        setstate(state_before)
    else:
        receiver.__dict__.update(state_before)
    return receiver


def _call(
    original: Callable[..., Any], args: list, kwargs: dict
) -> tuple[Any, Exception | None]:
    """Call ``original`` with recording suppressed, capturing an ordinary exception.

    A ``BaseException`` that is not an ``Exception`` (``KeyboardInterrupt``) propagates:
    it stops the replay, it is not the function's behavior.
    """
    with suppressed():
        try:
            return original(*args, **kwargs), None
        except Exception as e:  # noqa: BLE001 - what the call raised is its result
            return None, e


def _capped(result: ComparisonResult, max_diffs: int) -> tuple[ComparisonResult, int]:
    """Keep at most ``max_diffs`` diffs, so a report over many records stays small."""
    dropped = max(len(result.diffs) - max_diffs, 0)
    if not dropped:
        return result, 0
    return ComparisonResult(result.passed, result.diffs[:max_diffs]), dropped
