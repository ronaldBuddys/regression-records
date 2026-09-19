"""Comparison of polars DataFrames and Series: schema, then length, then values."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import polars as pl

from .base import (
    CompareFn,
    Diff,
    ToleranceInfo,
    compare_columns,
    describe_positions,
    require_both,
    safe_repr,
    schema_message,
    schemas_match,
)
from .numpy_comparator import DEFAULT_MAX_DIFFS, compare_arrays, mismatch_mask
from .tolerance import ToleranceConfig

# Recent polars deprecates `explode()` without saying what an empty list becomes; older
# versions do not know the argument. Only non-empty lists are ever exploded here, so
# the value is immaterial - it is passed to keep the warning out of callers' test runs.
_EXPLODE_KWARGS: dict[str, bool] = (
    {"empty_as_null": False}
    if "empty_as_null" in inspect.signature(pl.Series.explode).parameters
    else {}
)


def _is_tolerant_numeric(dtype: pl.DataType) -> bool:
    """Whether a column is compared within tolerance rather than exactly.

    Integers and floats only. ``Decimal`` is numeric to polars, but it is exact by
    construction and reaches numpy as Python objects, so it compares exactly.
    """
    return dtype.is_integer() or dtype.is_float()


def _zero_where(series: pl.Series, mask: np.ndarray) -> np.ndarray:
    """``series`` as a numpy array of its own dtype, zero at nulls and under ``mask``.

    ``to_numpy()`` on a column with nulls goes through float64 to hold them as NaN,
    which makes a null look like a NaN and rounds integers beyond 2**53. Filling first
    keeps the dtype; the nulls themselves are compared separately, as masks.
    """
    return np.where(mask, 0, series.fill_null(0).to_numpy())


def _compare_nullness(
    expected: pl.Series,
    actual: pl.Series,
    path: str,
    diffs: list[Diff],
    max_diffs: int,
) -> np.ndarray:
    """Append a diff if the two differ in *where* they are null; return those rows."""
    differing = (expected.is_null() != actual.is_null()).to_numpy()
    if differing.any():
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=safe_repr(
                    expected.filter(differing).head(max_diffs).to_list()
                ),
                actual=safe_repr(actual.filter(differing).head(max_diffs).to_list()),
                message=(
                    f"{int(differing.sum())} of {expected.len()} value(s) are null on "
                    f"one side only at row(s): {describe_positions(differing, max_diffs)}"
                ),
            )
        )
    return differing


def _has_float_leaf(dtype: pl.DataType) -> bool:
    """Whether a dtype is, or nests, a float: the columns tolerance has a say in."""
    if isinstance(dtype, (pl.List, pl.Array)):
        return _has_float_leaf(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_has_float_leaf(field.dtype) for field in dtype.fields)
    return dtype.is_float()


def _lengths(series: pl.Series) -> pl.Series:
    """Per-row element counts of a ``List`` or ``Array`` series (null for a null row)."""
    if isinstance(series.dtype, pl.Array):
        return series.arr.to_list().list.len()
    return series.list.len()


def _flag_nested_rows(
    expected: pl.Series,
    actual: pl.Series,
    rows: np.ndarray,
    tolerance: ToleranceConfig,
    differing: np.ndarray,
) -> None:
    """Set ``differing[row]`` for each outer row in which two nested series differ.

    Walks a ``List``/``Array``/``Struct`` dtype down to its leaves, so that the
    numeric ones can be compared within tolerance like a flat column is.

    Parameters
    ----------
    expected, actual : polars.Series
        Same dtype and length: a column, or what is left of one after flattening.
    rows : numpy.ndarray
        For each element of ``expected``, the row of the original column it came from.
    tolerance : ToleranceConfig
        Applied to integer and float leaves; every other leaf compares exactly.
    differing : numpy.ndarray
        Boolean mask over the original column's rows, updated in place.
    """
    dtype = expected.dtype
    either_null = (expected.is_null() | actual.is_null()).to_numpy()
    differing[rows[(expected.is_null() != actual.is_null()).to_numpy()]] = True

    if _is_tolerant_numeric(dtype):
        mismatched = mismatch_mask(
            _zero_where(expected, either_null),
            _zero_where(actual, either_null),
            tolerance,
        )
        differing[rows[mismatched]] = True
    elif isinstance(dtype, pl.Struct):
        for field in dtype.fields:
            _flag_nested_rows(
                expected.struct.field(field.name),
                actual.struct.field(field.name),
                rows,
                tolerance,
                differing,
            )
    elif isinstance(dtype, (pl.List, pl.Array)):
        lengths = _lengths(expected).fill_null(0).to_numpy()
        same_length = lengths == _lengths(actual).fill_null(0).to_numpy()
        differing[rows[~either_null & ~same_length]] = True
        # Only non-empty lists of matching length are flattened, so element i of one
        # side is the counterpart of element i of the other, and neither a null nor an
        # empty list takes up a slot.
        keep = ~either_null & same_length & (lengths > 0)
        _flag_nested_rows(
            expected.filter(keep).explode(**_EXPLODE_KWARGS),
            actual.filter(keep).explode(**_EXPLODE_KWARGS),
            np.repeat(rows[keep], lengths[keep]),
            tolerance,
            differing,
        )
    else:
        differing[rows[(~expected.eq_missing(actual)).to_numpy()]] = True


def _compare_nested_values(
    expected: pl.Series,
    actual: pl.Series,
    path: str,
    tolerance: ToleranceConfig,
    diffs: list[Diff],
    max_diffs: int,
    already_reported: np.ndarray,
) -> None:
    """Compare a nested column holding floats, applying tolerance to its leaves."""
    differing = np.zeros(expected.len(), dtype=bool)
    _flag_nested_rows(expected, actual, np.arange(expected.len()), tolerance, differing)
    differing &= ~already_reported
    if not differing.any():
        return
    diffs.append(
        Diff(
            path=path,
            kind="value_mismatch",
            expected=safe_repr(expected.filter(differing).head(max_diffs).to_list()),
            actual=safe_repr(actual.filter(differing).head(max_diffs).to_list()),
            message=(
                f"{int(differing.sum())} of {expected.len()} nested value(s) differ "
                f"beyond tolerance at row(s): {describe_positions(differing, max_diffs)}"
            ),
            tolerance_used=ToleranceInfo(rtol=tolerance.rtol, atol=tolerance.atol),
        )
    )


def _compare_values(
    expected: pl.Series,
    actual: pl.Series,
    path: str,
    tolerance: ToleranceConfig,
    diffs: list[Diff],
    max_diffs: int,
) -> None:
    """Compare two same-length series' values: where they are null, then the rest.

    A null is missing and a NaN is a number, so one becoming the other is reported,
    whatever ``equal_nan`` says; that setting governs NaN against NaN only. Rows
    reported for their nullness are not reported again for their values.
    """
    # `equals` counts NaN as equal to NaN, which is not the caller's view if they
    # turned equal_nan off.
    nan_is_a_difference = _has_float_leaf(expected.dtype) and not tolerance.equal_nan
    if not nan_is_a_difference and expected.equals(actual):
        return

    null_differs = _compare_nullness(expected, actual, path, diffs, max_diffs)

    if _is_tolerant_numeric(expected.dtype) and _is_tolerant_numeric(actual.dtype):
        # Dtypes already matched via the schema check, so don't re-report them here.
        either_null = (expected.is_null() | actual.is_null()).to_numpy()
        compare_arrays(
            _zero_where(expected, either_null),
            _zero_where(actual, either_null),
            path,
            tolerance,
            diffs,
            check_dtype=False,
            max_diffs=max_diffs,
        )
        return

    if expected.dtype == actual.dtype and _has_float_leaf(expected.dtype):
        _compare_nested_values(
            expected, actual, path, tolerance, diffs, max_diffs, null_differs
        )
        return

    try:
        differing = (~expected.eq_missing(actual)).to_numpy() & ~null_differs
    except Exception:  # noqa: BLE001 - dtypes polars can't compare elementwise
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
            expected=safe_repr(expected.filter(differing).head(max_diffs).to_list()),
            actual=safe_repr(actual.filter(differing).head(max_diffs).to_list()),
            message=(
                f"{int(differing.sum())} of {expected.len()} value(s) differ at row(s): "
                f"{describe_positions(differing, max_diffs)}"
            ),
        )
    )


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
        """Compare two frames: schema and column order, then height, then values.

        The first level that differs is reported and ends the comparison, as does a
        type mismatch when only one side is a dataframe.
        """
        if not require_both(expected, actual, pl.DataFrame, path, diffs):
            return

        expected_schema = dict(expected.schema)
        actual_schema = dict(actual.schema)
        # Order is checked whatever ``strict_dtype`` says: it relaxes dtypes, and a
        # frame whose columns moved is a different frame to anything positional.
        if (
            not schemas_match(expected_schema, actual_schema, tolerance.strict_dtype)
            or expected.columns != actual.columns
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
            lambda name, column_path, column_tol: _compare_values(
                expected.get_column(name),
                actual.get_column(name),
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


class PolarsSeriesComparator:
    """Compares ``polars.Series`` values (name, dtype, length, then values)."""

    def __init__(self, max_diffs: int = DEFAULT_MAX_DIFFS) -> None:
        self.max_diffs = max_diffs

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Claim the pair when either side is a ``polars.Series``."""
        return isinstance(expected, pl.Series) or isinstance(actual, pl.Series)

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare two series: name and dtype, then the length, then the values.

        The dtype is checked only under ``strict_dtype``. The first level that differs
        is reported and ends the comparison, as does a type mismatch when only one side
        is a series.
        """
        if not require_both(expected, actual, pl.Series, path, diffs):
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

        if expected.len() != actual.len():
            diffs.append(
                Diff(
                    path=path,
                    kind="length_mismatch",
                    expected=str(expected.len()),
                    actual=str(actual.len()),
                    message=f"series has {actual.len()} value(s), expected {expected.len()}",
                )
            )
            return

        _compare_values(expected, actual, path, tolerance, diffs, self.max_diffs)
