"""Structured comparison of a recorded value against a freshly-computed one.

:func:`compare` walks both values in lockstep, collecting *every* difference it finds
(it never short-circuits on the first one) into a :class:`ComparisonResult`. It is a
pure function and never raises on a mismatch - turning a failed comparison into a
failed case is :func:`~regression_records.replay`'s job, which keeps comparison logic
independent of recording and replaying.

Containers (dict/list/tuple/set) and scalars are walked here; "inconvenient" leaves
(numpy arrays, dataframes, user-defined objects) are delegated to a
:class:`~regression_records.comparison.base.Comparator` from a
:class:`~regression_records.comparison.base.ComparatorRegistry`, mirroring how
serialization dispatches to its serializer registry.
"""

from __future__ import annotations

from typing import Any

from .._object_state import is_namedtuple
from .._optional import library_is_absent
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
    safely_equal,
    type_name,
)
from .exception_diffs import compare_exceptions, exception_type_name
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
    "exception_type_name",
    "register_comparator",
]


def _sequence_kind(value: Any) -> type | None:
    """Return the type a sequence is compared as, or ``None`` if it is not one.

    That is the type it comes back from a record as. A named tuple is recorded with its
    class, so it is its own kind; any other ``list`` or ``tuple`` subclass is recorded
    as the builtin - as a ``dict`` or ``int`` subclass is - and so compares as one.
    Comparing exact types instead made every record of such a value fail forever.
    """
    if is_namedtuple(value):
        return type(value)
    for builtin in (list, tuple):
        if isinstance(value, builtin):
            return builtin
    return None


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
        if _sequence_kind(expected) is not _sequence_kind(actual):
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
        for position, (e, a) in enumerate(zip(expected, actual, strict=True)):
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
        if tol.equal_nan and _same_nan(expected, actual):
            return
        compare_exactly(expected, actual, path, self.diffs)


def _same_nan(expected: Any, actual: Any) -> bool:
    """Whether both values are the same NaN-like value, which ``==`` never admits.

    NaN-like means not equal to itself: ``Decimal("NaN")``, a ``complex`` with a NaN
    part, ``pandas.NaT``. Matching ``repr`` keeps the rest of the value compared - a
    complex's other part, a decimal NaN's payload.
    """
    return (
        type(expected) is type(actual)
        and not safely_equal(expected, expected)
        and not safely_equal(actual, actual)
        and safe_repr(expected) == safe_repr(actual)
    )


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

    ``root`` names the value being compared in reported paths. Replay passes
    ``"$.output"``, ``"$.state_before"``, and ``"$.state_after"`` so a single record's
    diffs stay distinguishable, and so ``ToleranceConfig.path_overrides`` can target
    each part separately.

    Parameters
    ----------
    expected : Any
        The reference value - what a record holds.
    actual : Any
        The freshly-produced value to check against it.
    tolerance : ToleranceConfig, optional
        Numeric comparison tolerances; ``ToleranceConfig()`` when omitted. Its
        ``rtol`` is relative to ``expected``, so swapping the two sides can change the
        verdict.
    registry : ComparatorRegistry, optional
        Comparators consulted, first match wins, before the built-in container and
        scalar handling; :data:`default_comparator_registry` when omitted.
    root : str, default "$"
        The path given to the top-level value; every reported ``Diff.path`` starts with
        it.

    Returns
    -------
    ComparisonResult
        Passed iff no difference was found; otherwise ``diffs`` lists every one, in the
        order the walk met them.

    Notes
    -----
    A mismatch is reported, never raised. A ``dict`` is compared key by key, a
    ``list`` or ``tuple`` element by element (the two are not interchangeable, and a
    length mismatch is reported without descending), a ``set`` by exact membership, and
    numbers - ``bool`` excluded - within ``tolerance``. Anything else must agree in
    type and compare equal.

    Examples
    --------
    >>> from regression_records.comparison import ToleranceConfig, compare
    >>> result = compare({"price": 100.0, "qty": 3}, {"price": 100.5, "qty": 3})
    >>> result.passed
    False
    >>> [(d.path, d.kind) for d in result.diffs]
    [('$.price', 'value_mismatch')]
    >>> bool(compare({"price": 100.0}, {"price": 100.5}, ToleranceConfig(rtol=0.01)))
    True
    >>> compare([1.0, 2.0], [1.0, 2.5], root="$.output").diffs[0].path
    '$.output[1]'
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
        from .polars_comparator import PolarsDataFrameComparator, PolarsSeriesComparator
    except ImportError as e:
        if not library_is_absent(e, "polars"):
            raise
    else:
        registry.register(PolarsDataFrameComparator())
        registry.register(PolarsSeriesComparator())

    try:
        from .pandas_comparator import PandasDataFrameComparator, PandasSeriesComparator
    except ImportError as e:
        if not library_is_absent(e, "pandas"):
            raise
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

    Parameters
    ----------
    comparator : Comparator
        The comparator to add. It affects every comparison in the process that does not
        pass its own registry.
    priority : int, optional
        The list index to insert at (``0`` is consulted first). When omitted, the
        comparator goes after the ones already registered and before the fallback.

    See Also
    --------
    ComparatorRegistry.copy : Add a comparator for one replay only.
    """
    default_comparator_registry.register(comparator, priority=priority)
