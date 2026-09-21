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
    """Return ``repr(x)``, truncated to ``limit`` characters for use in a diff."""
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
    """
    if strict_dtype:
        return expected == actual
    return set(expected) == set(actual)


def schema_message(expected: dict, actual: dict) -> str:
    """Summarize how two ``{column: dtype}`` mappings differ.

    Shared by the dataframe comparators.
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
    """Records which tolerance was applied to a numeric ``Diff``, for diagnostics."""

    rtol: float
    atol: float
    max_abs_diff: float | None = None


@dataclass
class Diff:
    """One localized difference found while comparing an expected vs actual value.

    ``expected``/``actual`` hold short *reprs* (see :func:`safe_repr`), never the live
    objects - a ``Diff`` may outlive the comparison inside a ``RegressionMismatchError``
    and must not pin whole arrays or dataframes in memory.
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
_EXACT_FLOAT_INT = 2**53


def _is_big_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and abs(int(x)) > _EXACT_FLOAT_INT


def numbers_close(expected: float, actual: float, tol: ToleranceConfig) -> bool:
    """``numpy.isclose`` semantics, with NaN and infinity handled explicitly."""
    if math.isnan(expected) or math.isnan(actual):
        return tol.equal_nan and math.isnan(expected) and math.isnan(actual)
    if math.isinf(expected) or math.isinf(actual):
        return expected == actual
    return abs(actual - expected) <= tol.atol + tol.rtol * abs(expected)


def compare_numbers(
    expected: Any, actual: Any, path: str, tol: ToleranceConfig, diffs: list[Diff]
) -> None:
    """Append a diff if two numbers differ by more than ``tol`` allows."""
    if _is_big_int(expected) or _is_big_int(actual):
        # Beyond 2**53 a float cannot tell neighbouring integers apart, and beyond
        # ~1e308 float() raises OverflowError out of a function documented never to
        # raise. Such values are identifiers or exact counts, not measurements.
        if expected == actual:
            return
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=safe_repr(expected),
                actual=safe_repr(actual),
                message=f"{actual!r} != expected {expected!r} (compared exactly: too "
                "large for a float to represent)",
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
            del diffs[before:]
            diffs.append(
                Diff(
                    path=path,
                    kind="value_mismatch",
                    expected="<truncated>",
                    actual="<truncated>",
                    message=(
                        f"{len(names) - position} further column(s) starting at {name!r} "
                        f"also differ; reporting stopped after {max_diffs} differing columns"
                    ),
                )
            )
            return


@dataclass
class ComparisonResult:
    """The outcome of :func:`~regression_records.comparison.compare`.

    Truthy iff ``passed``. Never raises on its own - ``compare()`` is a pure
    function; deciding whether a failed comparison should raise is a decorator-layer
    policy (see ``RegressionMismatchError``).
    """

    passed: bool
    diffs: list[Diff] = field(default_factory=list)

    @classmethod
    def from_diffs(cls, diffs: list[Diff]) -> ComparisonResult:
        """Build a result that passes iff no diffs were found."""
        return cls(passed=not diffs, diffs=diffs)

    def __bool__(self) -> bool:
        """Return whether the comparison passed."""
        return self.passed

    def summary(self, max_diffs: int = 20) -> str:
        """Return a readable report listing at most ``max_diffs`` diffs, one a line."""
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
        """Return whether this comparator claims the ``expected``/``actual`` pair."""
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
        """Compare a claimed pair at ``path``, appending what differs to ``diffs``."""
        ...


class ComparatorRegistry:
    """An ordered collection of comparators, searched first-match-wins."""

    def __init__(self) -> None:
        self._comparators: list[Comparator] = []
        self._fallbacks = 0

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
        """
        if priority is None:
            priority = (
                len(self._comparators)
                if fallback
                else len(self._comparators) - self._fallbacks
            )
        self._comparators.insert(priority, comparator)
        if fallback:
            self._fallbacks += 1

    def copy(self) -> ComparatorRegistry:
        """Return a new registry with the same comparators, in the same order.

        The counterpart to :meth:`SerializerRegistry.copy`, and used the same way: copy,
        register, inject, so one decoration gains a comparator without every other
        decoration in the process gaining it too.
        """
        clone = ComparatorRegistry()
        clone._comparators = list(self._comparators)
        clone._fallbacks = self._fallbacks
        return clone

    def __len__(self) -> int:
        """Return the number of registered comparators, fallbacks included."""
        return len(self._comparators)

    def find(self, expected: Any, actual: Any) -> Comparator | None:
        """Return the first comparator claiming the pair, or ``None`` if none does."""
        for comparator in self._comparators:
            if comparator.can_handle(expected, actual):
                return comparator
        return None
