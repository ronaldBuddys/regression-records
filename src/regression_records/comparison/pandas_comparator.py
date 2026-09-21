"""Comparison of pandas DataFrames and Series: schema, index, then column values."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import (
    CompareFn,
    Diff,
    compare_columns,
    describe_positions,
    require_both,
    safe_repr,
    schema_message,
)
from .numpy_comparator import DEFAULT_MAX_DIFFS, compare_arrays
from .tolerance import ToleranceConfig


def _differing_mask(expected: pd.Series, actual: pd.Series) -> np.ndarray:
    """Positions where two same-length series differ, treating NaN == NaN as equal."""
    left = expected.to_numpy()
    right = actual.to_numpy()
    both_na = pd.isna(left) & pd.isna(right)
    return np.asarray(~((left == right) | both_na))


def _compare_index(
    expected: pd.Index, actual: pd.Index, path: str, diffs: list[Diff]
) -> bool:
    """Append a diff if the indexes differ; return True when they match."""
    if expected.equals(actual):
        return True
    kind = "length_mismatch" if len(expected) != len(actual) else "value_mismatch"
    diffs.append(
        Diff(
            path=f"{path}.index",
            kind=kind,
            expected=safe_repr(expected),
            actual=safe_repr(actual),
            message=f"index differs (expected length {len(expected)}, got {len(actual)})"
            if kind == "length_mismatch"
            else "index values differ",
        )
    )
    return False


def _compare_values(
    expected: pd.Series,
    actual: pd.Series,
    path: str,
    tolerance: ToleranceConfig,
    diffs: list[Diff],
    max_diffs: int,
) -> None:
    """Compare two same-length, same-dtype series' values."""
    if pd.api.types.is_numeric_dtype(expected) and pd.api.types.is_numeric_dtype(
        actual
    ):
        # Dtype was already checked at the schema level, so don't re-report it per column.
        compare_arrays(
            expected.to_numpy(),
            actual.to_numpy(),
            path,
            tolerance,
            diffs,
            check_dtype=False,
            max_diffs=max_diffs,
        )
        return

    differing = _differing_mask(expected, actual)
    if not differing.any():
        return
    diffs.append(
        Diff(
            path=path,
            kind="value_mismatch",
            expected=safe_repr(expected.to_numpy()[differing][:max_diffs]),
            actual=safe_repr(actual.to_numpy()[differing][:max_diffs]),
            message=(
                f"{int(differing.sum())} of {len(expected)} value(s) differ at position(s): "
                f"{describe_positions(differing, max_diffs)}"
            ),
        )
    )


class PandasDataFrameComparator:
    """Compares ``pandas.DataFrame`` values.

    Order of checks mirrors the polars comparator: a schema difference short-circuits
    value comparison, then the index, then per-column values. Reported positions are
    capped at ``max_diffs`` with full counts retained in each message.
    """

    def __init__(self, max_diffs: int = DEFAULT_MAX_DIFFS) -> None:
        self.max_diffs = max_diffs

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Claim the pair when either side is a ``pandas.DataFrame``."""
        return isinstance(expected, pd.DataFrame) or isinstance(actual, pd.DataFrame)

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare two frames: columns and dtypes, then the index, then each column.

        The first level that differs is reported and ends the comparison, as does a
        type mismatch when only one side is a dataframe.
        """
        if not require_both(expected, actual, pd.DataFrame, path, diffs):
            return

        expected_schema = {str(c): str(d) for c, d in expected.dtypes.items()}
        actual_schema = {str(c): str(d) for c, d in actual.dtypes.items()}
        if expected_schema != actual_schema or list(expected.columns) != list(
            actual.columns
        ):
            diffs.append(
                Diff(
                    path=path,
                    kind="schema_mismatch",
                    expected=safe_repr(expected_schema),
                    actual=safe_repr(actual_schema),
                    message=schema_message(expected_schema, actual_schema),
                )
            )
            return

        if not _compare_index(expected.index, actual.index, path, diffs):
            return

        compare_columns(
            list(expected.columns),
            lambda name, column_path, column_tol: _compare_values(
                expected[name],
                actual[name],
                column_path,
                column_tol,
                diffs,
                self.max_diffs,
            ),
            path,
            tolerance,
            self.max_diffs,
            diffs,
        )


class PandasSeriesComparator:
    """Compares ``pandas.Series`` values (name, dtype, index, then values)."""

    def __init__(self, max_diffs: int = DEFAULT_MAX_DIFFS) -> None:
        self.max_diffs = max_diffs

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Claim the pair when either side is a ``pandas.Series``."""
        return isinstance(expected, pd.Series) or isinstance(actual, pd.Series)

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare two series: name and dtype, then the index, then the values.

        The dtype is checked only under ``strict_dtype``. The first level that differs
        is reported and ends the comparison, as does a type mismatch when only one side
        is a series.
        """
        if not require_both(expected, actual, pd.Series, path, diffs):
            return

        if expected.name != actual.name or (
            tolerance.strict_dtype and expected.dtype != actual.dtype
        ):
            diffs.append(
                Diff(
                    path=path,
                    kind="schema_mismatch",
                    expected=f"name={expected.name!r}, dtype={expected.dtype}",
                    actual=f"name={actual.name!r}, dtype={actual.dtype}",
                    message="series name or dtype differs",
                )
            )
            return

        if not _compare_index(expected.index, actual.index, path, diffs):
            return
        _compare_values(expected, actual, path, tolerance, diffs, self.max_diffs)
