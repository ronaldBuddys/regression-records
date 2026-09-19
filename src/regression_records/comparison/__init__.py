"""Structured comparison of a recorded value against a freshly-computed one.

:func:`compare` walks both values in lockstep, collecting *every* difference it finds
(it never short-circuits on the first one) into a :class:`ComparisonResult`. It is a
pure function and never raises on a mismatch - deciding that a failed comparison should
become a ``RegressionMismatchError`` is the decorator layer's policy, which keeps
comparison logic independent of recording.

Containers (dict/list/tuple/set) and scalars are walked here; "inconvenient" leaves
(numpy arrays, dataframes, user-defined objects) are delegated to a
:class:`~regression_records.comparison.base.Comparator` from a
:class:`~regression_records.comparison.base.ComparatorRegistry`, mirroring how
serialization dispatches to its serializer registry.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import (
    Comparator,
    ComparatorRegistry,
    ComparisonResult,
    Diff,
    ToleranceInfo,
    compare_exactly,
    compare_numbers,
    is_number,
    safe_repr,
    type_name,
)
from .exception_diffs import compare_exceptions
from .paths import ROOT_PATH, child, index
from .tolerance import ToleranceConfig

__all__ = [
    "Comparator",
    "ComparatorRegistry",
    "ComparisonResult",
    "Diff",
    "ROOT_PATH",
    "ToleranceConfig",
    "ToleranceInfo",
    "compare",
    "compare_exceptions",
    "default_comparator_registry",
    "register_comparator",
]


def _type_mismatch(expected: Any, actual: Any, path: str, diffs: list[Diff]) -> None:
    diffs.append(
        Diff(
            path=path,
            kind="type_mismatch",
            expected=type_name(expected),
            actual=type_name(actual),
            message=f"expected {type_name(expected)}, got {type_name(actual)}",
        )
    )


class _Walker:
    """Carries the comparison's shared state (registry, tolerance, collected diffs).

    Comparators append their findings directly to ``self.diffs`` and recurse through
    ``self.walk``, which appends to the same list - so a comparator that both reports
    its own diffs and recurses can't double-count them.
    """

    def __init__(
        self, tolerance: ToleranceConfig, registry: ComparatorRegistry
    ) -> None:
        self.tolerance = tolerance
        self.registry = registry
        self.diffs: list[Diff] = []

    def walk(self, expected: Any, actual: Any, path: str) -> None:
        tol = self.tolerance.for_path(path)

        comparator = self.registry.find(expected, actual)
        if comparator is not None:
            comparator.compare(expected, actual, path, tol, self.walk, self.diffs)
            return

        if isinstance(expected, dict) or isinstance(actual, dict):
            self._walk_dict(expected, actual, path)
            return

        if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
            self._walk_sequence(expected, actual, path)
            return

        if isinstance(expected, (set, frozenset)) or isinstance(
            actual, (set, frozenset)
        ):
            self._walk_set(expected, actual, path)
            return

        self._walk_scalar(expected, actual, path, tol)

    def _walk_dict(self, expected: Any, actual: Any, path: str) -> None:
        if not (isinstance(expected, dict) and isinstance(actual, dict)):
            _type_mismatch(expected, actual, path, self.diffs)
            return
        for key, value in expected.items():
            if key in actual:
                self.walk(value, actual[key], child(path, key))
            else:
                self.diffs.append(
                    Diff(
                        path=child(path, key),
                        kind="missing_key",
                        expected=safe_repr(value),
                        actual="<absent>",
                        message=f"key {key!r} is missing",
                    )
                )
        for key, value in actual.items():
            if key not in expected:
                self.diffs.append(
                    Diff(
                        path=child(path, key),
                        kind="extra_key",
                        expected="<absent>",
                        actual=safe_repr(value),
                        message=f"unexpected key {key!r}",
                    )
                )

    def _walk_sequence(self, expected: Any, actual: Any, path: str) -> None:
        # list and tuple are deliberately not interchangeable: a function that started
        # returning a list where it used to return a tuple has changed its contract.
        if type(expected) is not type(actual) or not isinstance(
            expected, (list, tuple)
        ):
            _type_mismatch(expected, actual, path, self.diffs)
            return
        if len(expected) != len(actual):
            self.diffs.append(
                Diff(
                    path=path,
                    kind="length_mismatch",
                    expected=str(len(expected)),
                    actual=str(len(actual)),
                    message=f"length {len(actual)} != expected {len(expected)}",
                )
            )
            return
        for position, (e, a) in enumerate(zip(expected, actual)):
            self.walk(e, a, index(path, position))

    def _walk_set(self, expected: Any, actual: Any, path: str) -> None:
        if type(expected) is not type(actual) or not isinstance(
            expected, (set, frozenset)
        ):
            _type_mismatch(expected, actual, path, self.diffs)
            return
        # Set members have no stable position to compare pairwise, so membership is
        # exact - tolerance cannot meaningfully apply here.
        for missing in expected - actual:
            self.diffs.append(
                Diff(
                    path=path,
                    kind="missing_key",
                    expected=safe_repr(missing),
                    actual="<absent>",
                    message=f"set is missing member {missing!r}",
                )
            )
        for extra in actual - expected:
            self.diffs.append(
                Diff(
                    path=path,
                    kind="extra_key",
                    expected="<absent>",
                    actual=safe_repr(extra),
                    message=f"set has unexpected member {extra!r}",
                )
            )

    def _walk_scalar(
        self, expected: Any, actual: Any, path: str, tol: ToleranceConfig
    ) -> None:
        if is_number(expected) and is_number(actual):
            compare_numbers(expected, actual, path, tol, self.diffs)
            return
        compare_exactly(expected, actual, path, self.diffs)


def compare(
    expected: Any,
    actual: Any,
    tolerance: ToleranceConfig | None = None,
    *,
    registry: ComparatorRegistry | None = None,
    root: str = ROOT_PATH,
) -> ComparisonResult:
    """Compare a recorded ``expected`` value against a freshly-produced ``actual`` one.

    Collects all differences rather than stopping at the first, and never raises for a
    mismatch - inspect the returned :class:`ComparisonResult` (truthy iff it passed,
    with ``.diffs`` and ``.summary()`` for diagnostics).

    ``root`` names the value being compared in reported paths. The decorators pass
    ``"$.output"``, ``"$.state_before"``, and ``"$.state_after"`` so a single record's
    diffs stay distinguishable, and so ``ToleranceConfig.path_overrides`` can target
    each part separately.
    """
    walker = _Walker(
        tolerance or ToleranceConfig(), registry or default_comparator_registry
    )
    walker.walk(expected, actual, root)
    return ComparisonResult.from_diffs(walker.diffs)


def _build_default_registry() -> ComparatorRegistry:
    registry = ComparatorRegistry()

    from .numpy_comparator import NumpyArrayComparator, NumpyScalarComparator

    registry.register(NumpyArrayComparator())
    registry.register(NumpyScalarComparator())

    try:
        from .polars_comparator import PolarsDataFrameComparator
    except ImportError:
        pass
    else:
        registry.register(PolarsDataFrameComparator())

    try:
        from .pandas_comparator import PandasDataFrameComparator, PandasSeriesComparator
    except ImportError:
        pass
    else:
        registry.register(PandasDataFrameComparator())
        registry.register(PandasSeriesComparator())

    # Catch-all for user-defined objects. Registered as the fallback so it keeps last
    # refusal no matter what is registered afterwards.
    from .objects import GenericObjectComparator

    registry.register(GenericObjectComparator(), fallback=True)

    return registry


default_comparator_registry = _build_default_registry()


def register_comparator(comparator: Comparator, *, priority: int | None = None) -> None:
    """Register ``comparator`` on the module-level :data:`default_comparator_registry`.

    The comparator is inserted just before the generic object fallback, so it takes
    priority over that but not over the more specific built-in comparators - pass an
    explicit ``priority`` index to override even those.
    """
    default_comparator_registry.register(comparator, priority=priority)
