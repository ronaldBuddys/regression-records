"""Comparison of objects stored by their attributes, by recursing into those attributes."""

from __future__ import annotations

from typing import Any

from .._attributes import attributes
from .base import CompareFn, Diff, type_name


def compare_objects(
    expected: Any, actual: Any, path: str, compare_fn: CompareFn, diffs: list[Diff]
) -> None:
    """Compare two instances of one class attribute by attribute.

    The attributes are those :func:`~regression_records._attributes.attributes` reads,
    the same ones a record stores, handed back to the walker so nested arrays, frames
    and objects are compared by their own rules, with paths such as
    ``$.output.origin.x``. Objects are compared by their attributes, not by ``__eq__``:
    a class without one still diffs usefully, and one whose ``__eq__`` is loose is
    still checked strictly.

    Differing classes are a type mismatch.
    """
    if type(expected) is not type(actual):
        diffs.append(
            Diff(
                path=path,
                kind="type_mismatch",
                expected=type_name(expected),
                actual=type_name(actual),
                message=(
                    f"expected an instance of {type_name(expected)}, "
                    f"got {type_name(actual)}"
                ),
            )
        )
        return
    compare_fn(attributes(expected), attributes(actual), path)
