"""Comparison of polars DataFrames: schema first, then height, then column values."""

from __future__ import annotations

from typing import Any

import polars as pl

from .base import (
    CompareFn,
    Diff,
    compare_columns,
    describe_positions,
    require_both,
    safe_repr,
    schema_message,
    schemas_match,
)
from .numpy_comparator import DEFAULT_MAX_DIFFS, compare_arrays
from .tolerance import ToleranceConfig


class PolarsDataFrameComparator:
    """Compares ``polars.DataFrame`` values.

    A schema difference short-circuits value comparison: once columns or their dtypes
    disagree, per-row diffs are noise rather than signal. Reported per-column row
    indices are capped at ``max_diffs`` (the full count always appears in the message),
    and once more than ``max_diffs`` columns differ, a single summary diff replaces the
    remainder.
    """

    def __init__(self, max_diffs: int = DEFAULT_MAX_DIFFS) -> None:
        self.max_diffs = max_diffs

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Claim the pair when either side is a ``polars.DataFrame``."""
        return isinstance(expected, pl.DataFrame) or isinstance(actual, pl.DataFrame)

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare two frames: schema, then height, then each column's values.

        The first level that differs is reported and ends the comparison, as does a
        type mismatch when only one side is a dataframe.
        """
        if not require_both(expected, actual, pl.DataFrame, path, diffs):
            return

        expected_schema = dict(expected.schema)
        actual_schema = dict(actual.schema)
        if not schemas_match(expected_schema, actual_schema, tolerance.strict_dtype):
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

        if expected.height != actual.height:
            diffs.append(
                Diff(
                    path=path,
                    kind="length_mismatch",
                    expected=str(expected.height),
                    actual=str(actual.height),
                    message=f"dataframe has {actual.height} row(s), expected {expected.height}",
                )
            )
            return

        compare_columns(
            expected.columns,
            lambda name, column_path, column_tol: self._compare_column(
                expected.get_column(name),
                actual.get_column(name),
                column_path,
                column_tol,
                diffs,
            ),
            path,
            tolerance,
            self.max_diffs,
            diffs,
        )

    def _compare_column(
        self,
        expected: pl.Series,
        actual: pl.Series,
        path: str,
        tolerance: ToleranceConfig,
        diffs: list[Diff],
    ) -> None:
        if expected.equals(actual):
            return

        if expected.dtype.is_numeric():
            # Dtypes already matched via the schema check, so don't re-report them here.
            # Nulls surface as NaN through to_numpy(); `equal_nan` governs whether two
            # nulls in the same position count as equal.
            compare_arrays(
                expected.to_numpy(),
                actual.to_numpy(),
                path,
                tolerance,
                diffs,
                check_dtype=False,
                max_diffs=self.max_diffs,
            )
            return

        try:
            differing = (~expected.eq_missing(actual)).to_numpy()
        except Exception:  # nested List/Struct dtypes polars can't compare elementwise
            diffs.append(
                Diff(
                    path=path,
                    kind="value_mismatch",
                    expected=safe_repr(expected),
                    actual=safe_repr(actual),
                    message="column values differ (dtype does not support elementwise comparison)",
                )
            )
            return

        if not differing.any():
            return
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=safe_repr(
                    expected.filter(differing).head(self.max_diffs).to_list()
                ),
                actual=safe_repr(
                    actual.filter(differing).head(self.max_diffs).to_list()
                ),
                message=(
                    f"{int(differing.sum())} of {expected.len()} value(s) differ at row(s): "
                    f"{describe_positions(differing, self.max_diffs)}"
                ),
            )
        )
