"""Comparison of numpy arrays, plus the shared array-diffing helper reused by the
dataframe comparators (which compare column-by-column as numpy arrays)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import CompareFn, Diff, ToleranceInfo, describe_positions, require_both, safe_repr
from .tolerance import ToleranceConfig

#: Numpy dtype kinds whose values are compared with rtol/atol rather than exactly.
_NUMERIC_KINDS = frozenset("biufc")

DEFAULT_MAX_DIFFS = 20


def compare_arrays(
    expected: np.ndarray,
    actual: np.ndarray,
    path: str,
    tol: ToleranceConfig,
    diffs: list[Diff],
    *,
    check_dtype: bool = True,
    max_diffs: int = DEFAULT_MAX_DIFFS,
) -> None:
    """Append diffs describing how ``actual`` differs from ``expected``.

    Shared by :class:`NumpyArrayComparator` and the polars/pandas comparators. Set
    ``check_dtype=False`` where dtype has already been checked at a higher level (a
    dataframe's schema), so a single dtype change isn't reported once per column.
    """
    if expected.shape != actual.shape:
        diffs.append(
            Diff(
                path=path,
                kind="shape_mismatch",
                expected=str(expected.shape),
                actual=str(actual.shape),
                message=f"array shape {actual.shape} != expected {expected.shape}",
            )
        )
        return

    numeric = expected.dtype.kind in _NUMERIC_KINDS and actual.dtype.kind in _NUMERIC_KINDS
    if check_dtype and expected.dtype != actual.dtype and tol.strict_dtype:
        diffs.append(
            Diff(
                path=path,
                kind="dtype_mismatch",
                expected=str(expected.dtype),
                actual=str(actual.dtype),
                message=f"array dtype {actual.dtype} != expected {expected.dtype}",
            )
        )
        if not numeric:
            # Values of incomparable kinds (e.g. float vs str) - the dtype diff is the
            # whole story, and elementwise comparison below would be meaningless.
            return

    if numeric:
        close = np.isclose(actual, expected, rtol=tol.rtol, atol=tol.atol, equal_nan=tol.equal_nan)
        mismatched = ~close
        if not mismatched.any():
            return
        max_abs_diff = _max_abs_diff(expected, actual, mismatched)
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=safe_repr(expected[mismatched][:max_diffs]),
                actual=safe_repr(actual[mismatched][:max_diffs]),
                message=(
                    f"{int(mismatched.sum())} of {expected.size} element(s) differ beyond "
                    f"tolerance at index/indices: {describe_positions(mismatched, max_diffs)}"
                ),
                tolerance_used=ToleranceInfo(
                    rtol=tol.rtol, atol=tol.atol, max_abs_diff=max_abs_diff
                ),
            )
        )
        return

    mismatched = np.asarray(expected != actual)
    if not mismatched.any():
        return
    diffs.append(
        Diff(
            path=path,
            kind="value_mismatch",
            expected=safe_repr(expected[mismatched][:max_diffs]),
            actual=safe_repr(actual[mismatched][:max_diffs]),
            message=(
                f"{int(mismatched.sum())} of {expected.size} element(s) differ at "
                f"index/indices: {describe_positions(mismatched, max_diffs)}"
            ),
        )
    )


def _max_abs_diff(
    expected: np.ndarray, actual: np.ndarray, mismatched: np.ndarray
) -> Optional[float]:
    """Largest finite absolute difference among mismatched elements, if measurable."""
    try:
        delta = np.abs(actual[mismatched].astype(float) - expected[mismatched].astype(float))
    except (TypeError, ValueError):  # pragma: no cover - complex/unsubtractable dtypes
        return None
    finite = delta[np.isfinite(delta)]
    return float(finite.max()) if finite.size else None


class NumpyArrayComparator:
    """Compares ``numpy.ndarray`` values with shape, dtype, and tolerance awareness."""

    def __init__(self, max_diffs: int = DEFAULT_MAX_DIFFS) -> None:
        self.max_diffs = max_diffs

    def can_handle(self, expected: Any, actual: Any) -> bool:
        return isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray)

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        if not require_both(
            expected, actual, np.ndarray, path, diffs, note=" (only one side is a numpy array)"
        ):
            return
        compare_arrays(expected, actual, path, tolerance, diffs, max_diffs=self.max_diffs)
