"""Comparison of arbitrary user-defined objects, by recursing into their state."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np

from .._object_state import object_state
from .base import CompareFn, Diff, safely_equal, type_name
from .tolerance import ToleranceConfig

# Types that must never be routed through state-based comparison: builtins and simple
# value types compare perfectly well with ``==``, and arrays/scalars have their own
# comparator. Everything else is a candidate, since every object answers
# ``__getstate__`` - so this exclusion is the whole of the check below.
#
# This is not redundant with registry order. The walker consults the registry *before*
# its own structural branches, deliberately, so that a user can register a comparator
# for a dict or list subclass and have it win. That is what makes the exclusions
# load-bearing here rather than something position-in-the-registry could express.
_EXCLUDED = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    dict,
    list,
    tuple,
    set,
    frozenset,
    datetime.date,  # also covers datetime.datetime
    datetime.time,
    datetime.timedelta,
    np.ndarray,
    np.generic,
)


def is_custom_object(obj: Any) -> bool:
    """Decide whether ``obj`` is compared by recursing into its attribute state.

    The generic serializer is a fallback that claims every object, so this is every
    object too - minus the types this walker handles more specifically. Anything
    recordable as a generic object is therefore also compared as one.
    """
    return not isinstance(obj, _EXCLUDED)


class GenericObjectComparator:
    """Compares two instances of the same user-defined class attribute-by-attribute.

    State is taken from ``__getstate__()`` when the class defines one, else from
    ``__dict__`` - the same source :class:`GenericObjectSerializer` records - and then
    handed back to the top-level walker, so nested arrays/dataframes/objects are
    compared by their own comparators with correct paths and tolerances.
    """

    def can_handle(self, expected: Any, actual: Any) -> bool:
        """Claim the pair when either side passes :func:`is_custom_object`."""
        return is_custom_object(expected) or is_custom_object(actual)

    def compare(
        self,
        expected: Any,
        actual: Any,
        path: str,
        tolerance: ToleranceConfig,
        compare_fn: CompareFn,
        diffs: list[Diff],
    ) -> None:
        """Compare two objects of the same class by recursing into their state.

        Differing classes are a type mismatch; objects exposing no inspectable state
        fall back to plain equality.
        """
        if type(expected) is not type(actual):
            diffs.append(
                Diff(
                    path=path,
                    kind="type_mismatch",
                    expected=type_name(expected),
                    actual=type_name(actual),
                    message=f"expected an instance of {type_name(expected)}, got {type_name(actual)}",
                )
            )
            return

        expected_state = object_state(expected)
        actual_state = object_state(actual)
        if expected_state is None or actual_state is None:
            _compare_opaque(expected, actual, path, diffs)
            return

        compare_fn(expected_state, actual_state, path)


def _compare_opaque(expected: Any, actual: Any, path: str, diffs: list[Diff]) -> None:
    """Last resort for objects with no inspectable state: plain equality."""
    if not safely_equal(expected, actual):
        diffs.append(
            Diff(
                path=path,
                kind="value_mismatch",
                expected=type_name(expected),
                actual=type_name(actual),
                message=(
                    "objects are unequal and expose no inspectable state to diff; "
                    "register a custom Comparator for this type for better diagnostics"
                ),
            )
        )
