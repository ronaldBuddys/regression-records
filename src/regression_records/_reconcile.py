"""Reconcile a fresh call's result against the record of an earlier one.

One definition of "does this call still behave as recorded", used by replay.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .comparison import ComparisonResult, compare, compare_exceptions
from .comparison.base import ComparatorRegistry, Diff
from .comparison.paths import OUTPUT_ROOT, STATE_AFTER_ROOT, STATE_BEFORE_ROOT
from .comparison.tolerance import ToleranceConfig
from .records import RegressionRecord
from .serialization import DeserializationContext


def reconcile(
    recorded: RegressionRecord,
    value: Any,
    exception: BaseException | None,
    deser_ctx: DeserializationContext,
    *,
    tolerance: ToleranceConfig | None = None,
    comparator_registry: ComparatorRegistry | None = None,
    states: Mapping[str, Any] | None = None,
) -> ComparisonResult:
    """Compare a fresh call's outcome, and optionally its state, against ``recorded``.

    Parameters
    ----------
    recorded : RegressionRecord
        The record of the earlier call.
    value : Any
        What the fresh call returned; ignored when ``exception`` is given.
    exception : BaseException or None
        What the fresh call raised, or None if it returned.
    deser_ctx : DeserializationContext
        Resolves the record's side files.
    tolerance : ToleranceConfig, optional
        Numeric tolerances; the defaults when omitted.
    comparator_registry : ComparatorRegistry, optional
        The comparators to use; the default registry when omitted.
    states : Mapping[str, Any], optional
        Fresh receiver snapshots keyed by their root, ``STATE_BEFORE_ROOT`` and/or
        ``STATE_AFTER_ROOT``. Each is compared against the recorded snapshot of the same
        name, when the record holds one; a root left out is not compared.

    Returns
    -------
    ComparisonResult
        Every difference, under ``$.output``, ``$.state_before`` and ``$.state_after``.

    Raises
    ------
    DeserializationError, IdentityResolutionError
        If a recorded value cannot be loaded.
    """
    recorded_states = {
        STATE_BEFORE_ROOT: recorded.state_before,
        STATE_AFTER_ROOT: recorded.state_after,
    }
    diffs = _compare_result(
        recorded, value, exception, deser_ctx, tolerance, comparator_registry
    )
    for root, fresh_state in (states or {}).items():
        recorded_state = recorded_states[root]
        if recorded_state is None:
            continue
        diffs.extend(
            compare(
                recorded_state.load(deser_ctx),
                fresh_state,
                tolerance,
                registry=comparator_registry,
                root=root,
            ).diffs
        )
    return ComparisonResult.from_diffs(diffs)


def _compare_result(
    recorded: RegressionRecord,
    value: Any,
    exception: BaseException | None,
    deser_ctx: DeserializationContext,
    tolerance: ToleranceConfig | None,
    comparator_registry: ComparatorRegistry | None,
) -> list[Diff]:
    """Compare the return value, or the exception raised in its place."""
    if recorded.succeeded and exception is None:
        return compare(
            recorded.load_output(deser_ctx),
            value,
            tolerance,
            registry=comparator_registry,
            root=OUTPUT_ROOT,
        ).diffs

    # At least one side raised, so there is no output to compare - the exception (or its
    # absence) is the whole result.
    return compare_exceptions(
        recorded.exception.exc_type if not recorded.succeeded else None,
        exception,
        root=OUTPUT_ROOT,
    ).diffs
