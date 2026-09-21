"""The outer steps every dataframe comparator takes, whatever the library.

A pandas and a polars frame are compared the same way from the outside: both sides must
be frames, then the schema and column order must agree, then the rows, then each column
is compared under its own resolved tolerance. Only how each step reads a frame differs,
so a library supplies that through a few small hooks. Keeping the steps in one place is
what keeps the libraries consistent: when they were written twice, one honoured
``strict_dtype`` and the other did not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .base import (
    CompareFn,
    Diff,
    compare_columns,
    require_both,
    safe_repr,
    schema_message,
    schemas_match,
)
from .numpy_comparator import DEFAULT_MAX_DIFFS
from .tolerance import ToleranceConfig


class FrameComparator(ABC):
    """Compares two dataframes: schema and column order, then rows, then each column.

    The first level that differs is reported and ends the comparison: once columns or
    their dtypes disagree, per-row diffs are noise rather than signal. Reported rows are
    capped at ``max_diffs`` per column (the full count is in each message), and once
    more than ``max_diffs`` columns differ, one summary diff replaces the rest.

    Parameters
    ----------
    max_diffs : int, default 20
        How many differing rows, and columns, to report in full.
    """

    #: The library's dataframe class, claimed on either side.
    frame_type: ClassVar[type]

    def __init__(self, max_diffs: int = DEFAULT_MAX_DIFFS) -> None:
        self.max_diffs = max_diffs

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Claim the pair when either side is this library's dataframe."""
        return isinstance(expected, self.frame_type) or isinstance(
            actual, self.frame_type
        )

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare two frames: schema and column order, then rows, then each column.

        A type mismatch, when only one side is a dataframe, is reported the same way.
        """
        if not require_both(expected, actual, self.frame_type, path, diffs):
            return

        expected_schema = self._schema(expected)
        actual_schema = self._schema(actual)
        # Order is checked whatever ``strict_dtype`` says: it relaxes dtypes, and a
        # frame whose columns moved is a different frame to anything positional.
        if not schemas_match(
            expected_schema, actual_schema, tolerance.strict_dtype
        ) or self._columns(expected) != self._columns(actual):
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

        if not self._rows_match(expected, actual, path, diffs):
            return

        compare_columns(
            self._columns(expected),
            lambda name, column_path, column_tol: self._compare_column(
                expected, actual, name, column_path, column_tol, diffs
            ),
            path,
            tolerance,
            self.max_diffs,
            diffs,
        )

    @abstractmethod
    def _schema(self, frame: Any) -> dict[str, Any]:
        """Return ``{column name: dtype}``, in column order, as the schema check sees it."""

    @abstractmethod
    def _columns(self, frame: Any) -> list[Any]:
        """Return the column labels, in order."""

    @abstractmethod
    def _rows_match(
        self, expected: Any, actual: Any, path: str, diffs: list[Diff]
    ) -> bool:
        """Append a diff and return False if the frames' rows do not line up."""

    @abstractmethod
    def _compare_column(
        self,
        expected: Any,
        actual: Any,
        name: Any,
        path: str,
        tolerance: ToleranceConfig,
        diffs: list[Diff],
    ) -> None:
        """Append whatever differs in column ``name`` to ``diffs``."""
