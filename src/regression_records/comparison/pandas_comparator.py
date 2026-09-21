"""Comparison of pandas DataFrames and Series: schema, index, then column values."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import (
    CompareFn,
    Diff,
    describe_positions,
    require_both,
    safe_repr,
)
from .frames import FrameComparator
from .numpy_comparator import DEFAULT_MAX_DIFFS, compare_arrays
from .tolerance import ToleranceConfig


def _differing_mask(expected: pd.Series, actual: pd.Series) -> np.ndarray:
    """Positions where two same-length series differ, treating NaN == NaN as equal."""
    # Through the extension arrays rather than to_numpy(): a nullable boolean or string
    # column becomes an object array holding pd.NA, where ``==`` yields NA and
    # ``.any()`` then raises "boolean value of NA is ambiguous".
    equal = pd.Series(expected.array == actual.array).fillna(False)
    both_na = expected.isna().to_numpy() & actual.isna().to_numpy()
    return ~(equal.to_numpy(dtype=bool) | both_na)


def _is_tolerant_numeric(series: pd.Series) -> bool:
    """Whether a column is compared within tolerance rather than exactly.

    Floats (and complex) only. Not booleans, though pandas counts them as numeric:
    ``True`` is not "1, within tolerance". Nor integers, nullable ones included: they
    are counts and identifiers, and compare exactly, as integer scalars do.
    """
    types = pd.api.types
    return (
        types.is_numeric_dtype(series)
        and not types.is_bool_dtype(series)
        and not types.is_integer_dtype(series)
    )


def _numeric_values(series: pd.Series) -> np.ndarray:
    """Return a numeric column as a numpy array of its numpy dtype.

    For a nullable (extension) column that is float64 with NaN where values are missing,
    as recent pandas returns from ``to_numpy()`` anyway. pandas 2.0 returned an object
    array holding ``pd.NA`` instead, which no tolerant comparison could be made on.
    """
    numpy_dtype = getattr(series.dtype, "numpy_dtype", None)
    if (
        not isinstance(series.dtype, pd.api.extensions.ExtensionDtype)
        or numpy_dtype is None
    ):
        return series.to_numpy()
    if series.isna().any():
        return series.to_numpy(dtype="float64", na_value=np.nan)
    return series.to_numpy(dtype=numpy_dtype)


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
    if _is_tolerant_numeric(expected) and _is_tolerant_numeric(actual):
        # Dtype was already checked at the schema level, so don't re-report it per column.
        compare_arrays(
            _numeric_values(expected),
            _numeric_values(actual),
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


def _is_default_text(dtype: Any) -> bool:
    """Whether ``dtype`` is what some pandas version holds plain strings in by default.

    That is ``object`` up to pandas 2 and ``str`` - the NaN-missing ``StringDtype`` -
    from pandas 3. The opt-in ``"string"`` dtype, missing as ``pd.NA``, is not one.
    """
    if pd.api.types.is_object_dtype(dtype):
        return True
    return (
        isinstance(dtype, pd.StringDtype) and getattr(dtype, "na_value", None) is np.nan
    )


def _dtype_label(dtype: Any) -> str:
    """Describe a column's dtype for the schema check.

    ``str()`` of a categorical dtype is just ``"category"``, which hid a change to the
    category set or to whether it is ordered. Categories are listed in order only when
    that order means something, as ``CategoricalDtype`` equality has it.

    ``object`` and ``str`` share a label. Parquet stores both as strings, and pandas 3
    reads any string column back as ``str`` (pandas 2 as ``object``), so a record of
    either could never pass against the other; the values are still compared.
    """
    if _is_default_text(dtype):
        return "str|object"
    if isinstance(dtype, pd.CategoricalDtype):
        categories = list(dtype.categories)
        if not dtype.ordered:
            categories.sort(key=repr)
        return f"category(ordered={dtype.ordered}, categories={categories!r})"
    return str(dtype)


class PandasDataFrameComparator(FrameComparator):
    """Compares ``pandas.DataFrame`` values.

    Rows line up when the indexes are equal; each column is then compared by position
    in the index. See ``comparison.frames.FrameComparator`` for the order of checks.
    """

    frame_type = pd.DataFrame

    def _schema(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {str(c): _dtype_label(d) for c, d in frame.dtypes.items()}

    def _columns(self, frame: pd.DataFrame) -> list[Any]:
        return list(frame.columns)

    def _rows_match(
        self, expected: pd.DataFrame, actual: pd.DataFrame, path: str, diffs: list[Diff]
    ) -> bool:
        return _compare_index(expected.index, actual.index, path, diffs)

    def _compare_column(
        self,
        expected: pd.DataFrame,
        actual: pd.DataFrame,
        name: Any,
        path: str,
        tolerance: ToleranceConfig,
        diffs: list[Diff],
    ) -> None:
        _compare_values(
            expected[name], actual[name], path, tolerance, diffs, self.max_diffs
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
            tolerance.strict_dtype
            and _dtype_label(expected.dtype) != _dtype_label(actual.dtype)
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
