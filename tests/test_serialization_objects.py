import numpy as np
import pytest

from regression_records.exceptions import SerializationError
from regression_records.serialization import to_json_tree

from ._helpers import round_trip


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y


class Widget:
    def __init__(self, name, value, tags=None):
        self._name = name
        self._value = value
        self._tags = tags or []

    def __getstate__(self):
        return {"name": self._name, "value": self._value, "tags": self._tags}

    def __setstate__(self, state):
        self._name = state["name"]
        self._value = state["value"]
        self._tags = state["tags"]

    def __eq__(self, other):
        return (
            isinstance(other, Widget)
            and self._name == other._name
            and self._value == other._value
            and self._tags == other._tags
        )


class Container:
    def __init__(self, point, arr):
        self.point = point
        self.arr = arr

    def __eq__(self, other):
        return (
            isinstance(other, Container)
            and self.point == other.point
            and np.array_equal(self.arr, other.arr)
        )


class Unstorable:
    def __init__(self):
        self.fn = lambda: None  # not JSON-representable, no serializer handles it


def test_dict_based_class_round_trip(ctx_pair):
    p = Point(1, 2)
    result = round_trip(p, ctx_pair)
    assert result == p
    assert type(result) is Point


def test_getstate_setstate_class_round_trip(ctx_pair):
    w = Widget("gadget", 42, tags=["a", "b"])
    result = round_trip(w, ctx_pair)
    assert result == w
    assert type(result) is Widget


def test_nested_custom_objects_round_trip(ctx_pair):
    c = Container(Point(1, 2), np.array([1, 2, 3]))
    result = round_trip(c, ctx_pair)
    assert result == c


def test_main_module_class_rejected(ctx_pair):
    ser_ctx, _ = ctx_pair
    p = Point(1, 2)
    original_module = Point.__module__
    Point.__module__ = "__main__"
    try:
        with pytest.raises(SerializationError, match="__main__"):
            to_json_tree(p, ser_ctx)
    finally:
        Point.__module__ = original_module


def test_unsupported_attribute_raises(ctx_pair):
    ser_ctx, _ = ctx_pair
    with pytest.raises(SerializationError):
        to_json_tree(Unstorable(), ser_ctx)
