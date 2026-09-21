"""The extensible comparator protocol, registry, and structured comparison result."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

import numpy as np

from .._registry import OrderedRegistry
from .paths import column
from .tolerance import ToleranceConfig

DiffKind = Literal[
    "type_mismatch",
    "value_mismatch",
    "shape_mismatch",
    "dtype_mismatch",
    "missing_key",
    "extra_key",
    "length_mismatch",
    "schema_mismatch",
    "exception_mismatch",
]


def type_name(x: Any) -> str:
    """Return a short, readable type name (module-qualified for non-builtins)."""
    t = type(x)
    return (
        t.__qualname__
        if t.__module__ == "builtins"
        else f"{t.__module__}.{t.__qualname__}"
    )


def safe_repr(x: Any, limit: int = 200) -> str:
    """Return ``repr(x)``, truncated to ``limit`` characters for use in a diff.

    Parameters
    ----------
    x : Any
        The value to describe.
    limit : int, default 200
        The longest string returned, counting the ``"..."`` that marks a truncation.

    Returns
    -------
    str
        The repr, or a ``<TypeName: repr() raised ...>`` placeholder if ``repr(x)``
        itself raises - this function never does.
    """
    try:
        r = repr(x)
    except Exception as e:  # noqa: BLE001 - compare() promises never to raise
        r = f"<{type(x).__name__}: repr() raised {type(e).__name__}>"
    return r if len(r) <= limit else r[: limit - 3] + "..."


def safely_equal(expected: Any, actual: Any) -> bool:
    """Evaluate ``expected == actual``, treating a raising or non-boolean ``__eq__`` as unequal.

    Comparison is a diagnostic, so a type with a hostile ``__eq__`` must produce a
    reported difference rather than an exception out of an otherwise pure function.
    """
    try:
        return bool(expected == actual)
    except Exception:  # noqa: BLE001 - any failure to compare means "not equal"
        return False


def describe_positions(mask: np.ndarray, limit: int) -> str:
    """Render the first ``limit`` True positions of a boolean mask as readable indices.

    Every comparator that reports "these elements differ" caps the positions it lists,
    and this is where that truncation format lives, so arrays, polars columns and pandas
    columns all describe a capped list the same way.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array, True where elements differ. Any number of dimensions.
    limit : int
        How many positions to list before summarizing the rest.

    Returns
    -------
    str
        Comma-separated positions in row-major order - a bare integer each for a 1-d
        mask, an index tuple otherwise - followed by ``", ... (N more)"`` when the mask
        holds more than ``limit``.
    """
    positions = np.argwhere(mask)
    shown = [tuple(int(i) for i in pos) for pos in positions[:limit]]
    rendered = ", ".join(str(p[0]) if len(p) == 1 else str(p) for p in shown)
    if len(positions) > limit:
        rendered += f", ... ({len(positions) - limit} more)"
    return rendered


def schemas_match(expected: dict, actual: dict, strict_dtype: bool) -> bool:
    """Whether two ``{column: dtype}`` mappings agree, as far as ``strict_dtype`` asks.

    With ``strict_dtype=False`` only the column names have to agree, which is what that
    setting already meant for arrays, numpy scalars and Series. One definition for both
    dataframe comparators, because the setting was honoured by neither.

    Parameters
    ----------
    expected, actual : dict
        ``{column name: dtype}``, in column order.
    strict_dtype : bool
        Whether dtypes must agree as well as names.

    Returns
    -------
    bool
        With ``strict_dtype``, whether the mappings are equal; without it, whether
        they have the same keys. Column order is not considered either way; the
        dataframe comparators check it themselves, whatever ``strict_dtype`` says.
    """
    if strict_dtype:
        return expected == actual
    return set(expected) == set(actual)


def schema_message(expected: dict, actual: dict) -> str:
    """Summarize how two ``{column: dtype}`` mappings differ.

    Shared by the dataframe comparators.

    Parameters
    ----------
    expected, actual : dict
        ``{column name: dtype}``, in column order.

    Returns
    -------
    str
        ``"schema mismatch: "`` followed by whichever of missing columns, unexpected
        columns and dtype changes apply, or ``"column order differs"`` when none does.
    """
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    changed = [
        f"{c}: {expected[c]} -> {actual[c]}"
        for c in expected
        if c in actual and expected[c] != actual[c]
    ]
    parts = []
    if missing:
        parts.append(f"missing column(s) {missing}")
    if extra:
        parts.append(f"unexpected column(s) {extra}")
    if changed:
        parts.append(f"dtype change(s) {changed}")
    if not parts:
        parts.append("column order differs")
    return "schema mismatch: " + "; ".join(parts)


@dataclass
class ToleranceInfo:
    """Records which tolerance was applied to a numeric ``Diff``, for diagnostics.

    Attributes
    ----------
    rtol : float
        The relative tolerance in force, as a fraction of the expected value.
    atol : float
        The absolute tolerance in force, in the units of the compared values.
    max_abs_diff : float or None, default None
        The largest finite ``abs(actual - expected)`` among the elements that failed,
        in the same units. None when it could not be measured, e.g. when every failing
        element involves a NaN or an infinity.
    """

    rtol: float
    atol: float
    max_abs_diff: float | None = None


@dataclass
class Diff:
    """One localized difference found while comparing an expected vs actual value.

    ``expected``/``actual`` hold short *reprs* (see :func:`safe_repr`), never the live
    objects - a ``Diff`` may outlive the comparison inside a ``RegressionMismatchError``
    and must not pin whole arrays or dataframes in memory.

    Attributes
    ----------
    path : str
        Where the difference is, in the grammar of
        :mod:`regression_records.comparison.paths` (``"$.output['price']"``).
    kind : str
        The category of difference: one of ``"type_mismatch"``, ``"value_mismatch"``,
        ``"shape_mismatch"``, ``"dtype_mismatch"``, ``"missing_key"``, ``"extra_key"``,
        ``"length_mismatch"``, ``"schema_mismatch"`` or ``"exception_mismatch"``.
    expected : str
        A short description of the recorded side - a truncated repr, a type name, a
        shape or a length, depending on ``kind``; ``"<absent>"`` for an extra key.
    actual : str
        The same for the freshly-produced side; ``"<absent>"`` for a missing key.
    message : str
        A one-line, human-readable account of the difference.
    tolerance_used : ToleranceInfo or None, default None
        The tolerance a numeric value failed against; None for every difference that
        is not a tolerant numeric comparison.
    """

    path: str
    kind: DiffKind
    expected: str
    actual: str
    message: str
    tolerance_used: ToleranceInfo | None = None


def is_number(x: Any) -> bool:
    """Whether ``x`` participates in tolerant numeric comparison.

    ``bool`` is deliberately excluded despite subclassing ``int``: ``True`` is not
    "1, within tolerance" - it is a different kind of answer.
    """
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(
        x, bool
    )


#: The largest magnitude up to which every integer is exactly a float.
EXACT_FLOAT_INT = 2**53


def _is_integer(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _is_big_int(x: Any) -> bool:
    return _is_integer(x) and abs(int(x)) > EXACT_FLOAT_INT


def numbers_close(expected: float, actual: float, tol: ToleranceConfig) -> bool:
    """``numpy.isclose`` semantics, with NaN and infinity handled explicitly.

    Parameters
    ----------
    expected : float
        The reference value; ``tol.rtol`` scales with its magnitude.
    actual : float
        The value being checked.
    tol : ToleranceConfig
        Supplies ``rtol``, ``atol`` and ``equal_nan``.

    Returns
    -------
    bool
        Whether ``abs(actual - expected) <= tol.atol + tol.rtol * abs(expected)``. Two
        NaNs are close iff ``tol.equal_nan``; a NaN and a number never are; an infinity
        is close only to the same infinity.
    """
    if math.isnan(expected) or math.isnan(actual):
        return tol.equal_nan and math.isnan(expected) and math.isnan(actual)
    if math.isinf(expected) or math.isinf(actual):
        return expected == actual
    return abs(actual - expected) <= tol.atol + tol.rtol * abs(expected)


def compare_numbers(
    expected: Any, actual: Any, path: str, tol: ToleranceConfig, diffs: list[Diff]
) -> None:
    """Append a diff if two numbers differ by more than ``tol`` allows.

    Parameters
    ----------
    expected, actual : int or float or numpy.integer or numpy.floating
        The recorded and the fresh number (see :func:`is_number`). Two integers are
        compared exactly, as is an integer beyond ``2**53`` in magnitude against a
        float, which can no longer tell such integers apart; otherwise they are
        compared as floats, within ``tol``.
    path : str
        The location reported on the diff.
    tol : ToleranceConfig
        The tolerances already resolved for ``path``.
    diffs : list[Diff]
        Appended to in place: one ``"value_mismatch"``, carrying a
        :class:`ToleranceInfo` when the comparison was a tolerant one.
    """
    both_integers = _is_integer(expected) and _is_integer(actual)
    if both_integers or _is_big_int(expected) or _is_big_int(actual):
        # Integers are counts and identifiers, not measurements: 1,000,000,050 rows is
        # not 1,000,000,000 "within tolerance". An integer against a float is compared
        # tolerantly, unless it is beyond 2**53, where a float cannot tell neighbours
        # apart - and beyond ~1e308 float() would raise OverflowError.
        if expected == actual:
            return
        reason = "integers" if both_integers else "too large for a float to represent"
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=safe_repr(expected),
                actual=safe_repr(actual),
                message=f"{actual!r} != expected {expected!r} (compared exactly: "
                f"{reason})",
            )
        )
        return
    if numbers_close(float(expected), float(actual), tol):
        return
    delta = abs(float(actual) - float(expected))
    diffs.append(
        Diff(
            path=path,
            kind="value_mismatch",
            expected=safe_repr(expected),
            actual=safe_repr(actual),
            message=f"{actual!r} differs from expected {expected!r} beyond tolerance",
            tolerance_used=ToleranceInfo(
                rtol=tol.rtol,
                atol=tol.atol,
                max_abs_diff=delta if math.isfinite(delta) else None,
            ),
        )
    )


def compare_exactly(expected: Any, actual: Any, path: str, diffs: list[Diff]) -> None:
    """Append a diff if two non-numeric values differ in type or in value."""
    if type(expected) is not type(actual):
        diffs.append(
            Diff(
                path=path,
                kind="type_mismatch",
                expected=type_name(expected),
                actual=type_name(actual),
                message=f"expected {type_name(expected)}, got {type_name(actual)}",
            )
        )
        return
    if not safely_equal(expected, actual):
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=safe_repr(expected),
                actual=safe_repr(actual),
                message=f"{actual!r} != expected {expected!r}",
            )
        )


def require_both(
    expected: Any,
    actual: Any,
    cls: type,
    path: str,
    diffs: list[Diff],
    *,
    note: str = "",
) -> bool:
    """Check that both sides are instances of ``cls``, recording a diff if not.

    A comparator is dispatched when *either* side matches its type, so each one has to
    handle the "only one side is a dataframe" case before it can do anything useful.

    Parameters
    ----------
    expected, actual : Any
        The pair the comparator was dispatched for.
    cls : type
        The type both must be instances of.
    path : str
        The location reported on the diff.
    diffs : list[Diff]
        Appended to in place: one ``"type_mismatch"`` when either side is not a
        ``cls``.
    note : str, default ""
        Text appended to the diff's message, for a comparator with something to add
        about why the types differ.

    Returns
    -------
    bool
        True when both sides are instances of ``cls`` and comparison can go on.
    """
    if isinstance(expected, cls) and isinstance(actual, cls):
        return True
    diffs.append(
        Diff(
            path=path,
            kind="type_mismatch",
            expected=type_name(expected),
            actual=type_name(actual),
            message=f"expected {type_name(expected)}, got {type_name(actual)}{note}",
        )
    )
    return False


def compare_columns(
    names: Sequence[Any],
    compare_one: Callable[[Any, str, ToleranceConfig], None],
    path: str,
    tolerance: ToleranceConfig,
    max_diffs: int,
    diffs: list[Diff],
) -> None:
    """Compare a dataframe's columns one at a time, capping how many are reported.

    ``compare_one(name, column_path, column_tolerance)`` appends its findings to
    ``diffs``. Once more than ``max_diffs`` columns have differed, the rest are replaced
    by a single summary diff: a frame where every column shifted should say so once, not
    once per column.

    The per-column tolerance is resolved here rather than by each caller, because a
    dataframe comparator walks its own columns instead of recursing through the shared
    dispatcher - so this is the only place a ``path_overrides`` entry naming a column
    (``"$.output['price']"``) can be honored, and doing it once means a new dataframe
    comparator cannot forget to.

    Parameters
    ----------
    names : Sequence[Any]
        The column labels to compare, in the order to report them.
    compare_one : Callable[[Any, str, ToleranceConfig], None]
        Called once per column as ``compare_one(name, column_path,
        column_tolerance)``; appends whatever differs in that column to ``diffs``.
    path : str
        The dataframe's path. Column paths are built from it with
        :func:`~regression_records.comparison.paths.column`.
    tolerance : ToleranceConfig
        The dataframe's tolerance, resolved per column with
        :meth:`ToleranceConfig.for_path`.
    max_diffs : int
        How many differing columns to report in full.
    diffs : list[Diff]
        Appended to in place. Past ``max_diffs`` differing columns, the next one's
        diffs are dropped for a single ``"value_mismatch"`` at ``path`` naming it and
        counting the columns after it, which are not compared.
    """
    differing = 0
    for position, name in enumerate(names):
        before = len(diffs)
        column_path = column(path, name)
        compare_one(name, column_path, tolerance.for_path(column_path))
        if len(diffs) == before:
            continue
        differing += 1
        if differing > max_diffs:
            skipped = len(names) - position - 1
            del diffs[before:]
            diffs.append(
                Diff(
                    path=path,
                    kind="value_mismatch",
                    expected="<truncated>",
                    actual="<truncated>",
                    message=(
                        f"column {name!r} also differs{_not_compared(skipped)}; "
                        f"reporting stopped after {max_diffs} differing columns"
                    ),
                )
            )
            return


def _not_compared(count: int) -> str:
    """Word the columns a truncated comparison skipped: they may or may not differ."""
    return f", and the {count} column(s) after it were not compared" if count else ""


@dataclass
class ComparisonResult:
    """The outcome of :func:`~regression_records.comparison.compare`.

    Truthy iff ``passed``. Never raises on its own - ``compare()`` is a pure
    function; deciding whether a failed comparison should raise is a decorator-layer
    policy (see ``RegressionMismatchError``).

    Attributes
    ----------
    passed : bool
        Whether the two values agreed within tolerance.
    diffs : list[Diff], default []
        Every difference found, in the order the walk met them; empty when ``passed``.
    """

    passed: bool
    diffs: list[Diff] = field(default_factory=list)

    @classmethod
    def from_diffs(cls, diffs: list[Diff]) -> ComparisonResult:
        """Build a result that passes iff no diffs were found.

        Parameters
        ----------
        diffs : list[Diff]
            The differences a comparison collected. Kept by reference, not copied.

        Returns
        -------
        ComparisonResult
            With ``passed`` set to whether ``diffs`` is empty.
        """
        return cls(passed=not diffs, diffs=diffs)

    def __bool__(self) -> bool:
        """Return whether the comparison passed.

        Returns
        -------
        bool
            ``passed``, so a result can be used directly in ``if`` and ``assert``.
        """
        return self.passed

    def summary(self, max_diffs: int = 20) -> str:
        """Return a readable report listing at most ``max_diffs`` diffs, one a line.

        Parameters
        ----------
        max_diffs : int, default 20
            How many diffs to list before summarizing the rest as ``... and N more``.

        Returns
        -------
        str
            ``"comparison passed"``, or a header line with the total number of diffs
            followed by one ``at <path> (<kind>): <message>`` line per listed diff.
            This is the message of a ``RegressionMismatchError``.
        """
        if self.passed:
            return "comparison passed"
        lines = [f"comparison failed with {len(self.diffs)} diff(s):"]
        for d in self.diffs[:max_diffs]:
            lines.append(f"  at {d.path!r} ({d.kind}): {d.message}")
        if len(self.diffs) > max_diffs:
            lines.append(f"  ... and {len(self.diffs) - max_diffs} more")
        return "\n".join(lines)


CompareFn = Callable[[Any, Any, str], None]


@runtime_checkable
class Comparator(Protocol):
    """Handles comparing one externally-representable type (numpy arrays, dataframes, ...).

    ``compare_fn(expected, actual, path)`` recurses back into the top-level dispatcher
    (for e.g. comparing a custom object's nested attributes); diffs found by ``self``
    or by ``compare_fn`` are appended directly to ``diffs``.
    """

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Return whether this comparator claims the ``expected``/``actual`` pair.

        Parameters
        ----------
        expected : Any
            The recorded value.
        actual : Any
            The freshly-produced value.

        Returns
        -------
        bool
            True to have :meth:`compare` called for this pair. Claim it when *either*
            side is the handled type, so that the other side being something else is
            reported as a type mismatch rather than falling through to a comparator
            that knows nothing about it.
        """
        ...

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare a claimed pair at ``path``, appending what differs to ``diffs``.

        Parameters
        ----------
        expected : Any
            The recorded value.
        actual : Any
            The freshly-produced value. Either side - not necessarily both - is of the
            handled type; see :func:`require_both`.
        path : str
            The pair's location, to put on every ``Diff`` and to extend (with the
            helpers in :mod:`regression_records.comparison.paths`) for nested values.
        tolerance : ToleranceConfig
            The tolerances already resolved for ``path``.
        compare_fn : Callable[[Any, Any, str], None]
            ``compare_fn(expected, actual, path)`` compares a nested pair with the full
            dispatcher, which resolves that path's tolerance and appends to the same
            ``diffs``.
        diffs : list[Diff]
            Appended to in place. Implementations report a difference here and should
            not raise for one.
        """
        ...


class ComparatorRegistry(OrderedRegistry["Comparator"]):
    """An ordered collection of comparators, searched first-match-wins.

    :meth:`copy` works as :meth:`SerializerRegistry.copy
    <regression_records.serialization.base.SerializerRegistry.copy>` does: copy,
    register, inject, so one decoration gains a comparator without every other
    decoration in the process gaining it too.
    """

    def register(
        self,
        comparator: Comparator,
        *,
        priority: int | None = None,
        fallback: bool = False,
    ) -> None:
        """Add ``comparator`` to the search order.

        By default it goes last among the ordinary comparators but *before* any
        registered fallback, so a catch-all registered with ``fallback=True`` keeps
        getting last refusal however many comparators are added after it. ``priority``
        is an explicit list index for the rare case that isn't good enough.

        Parameters
        ----------
        comparator : Comparator
            The comparator to add.
        priority : int, optional
            The list index to insert at (``0`` is consulted first). Ordinary entries
            precede fallbacks, and each must stay among its own kind. When omitted:
            the end of the list for a fallback, otherwise just before the first
            fallback.
        fallback : bool, default False
            Mark ``comparator`` as a catch-all that later default registrations are
            placed ahead of.

        Raises
        ------
        ValueError
            If ``priority`` would place an ordinary comparator among the fallbacks, a
            fallback among the ordinary ones, or is out of range.
        """
        self._insert(comparator, priority, fallback)

    def find(self, expected: Any, actual: Any) -> Comparator | None:
        """Return the first comparator claiming the pair, or ``None`` if none does.

        Parameters
        ----------
        expected : Any
            The recorded value.
        actual : Any
            The freshly-produced value.

        Returns
        -------
        Comparator or None
            The first comparator, in search order, whose ``can_handle(expected,
            actual)`` is true.
        """
        return self._first(lambda comparator: comparator.can_handle(expected, actual))
