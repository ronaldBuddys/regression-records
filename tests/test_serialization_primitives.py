import datetime
import json

import pytest

from regression_records.exceptions import SerializationError
from regression_records.serialization import to_json_tree

from ._helpers import round_trip


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 1, -5, 3.14, "", "hello", [1, 2, 3], {"a": 1, "b": [1, 2]}],
)
def test_primitive_round_trip(value, ctx_pair):
    assert round_trip(value, ctx_pair) == value


def test_nested_structure_round_trip(ctx_pair):
    value = {
        "a": [1, 2, {"b": (3, 4, None)}],
        "c": {"d": {"e": [True, False, "x"]}},
    }
    assert round_trip(value, ctx_pair) == value


def test_tuple_round_trip(ctx_pair):
    value = (1, "two", 3.0, (4, 5))
    result = round_trip(value, ctx_pair)
    assert result == value
    assert isinstance(result, tuple)
    assert isinstance(result[3], tuple)


def test_set_round_trip(ctx_pair):
    value = {1, 2, 3, "a"}
    result = round_trip(value, ctx_pair)
    assert result == value
    assert isinstance(result, set)


def test_set_serialization_is_deterministic(ctx_pair):
    ser_ctx, _ = ctx_pair
    value = {"z", "a", "m", "b"}
    tree1 = to_json_tree(value, ser_ctx)
    tree2 = to_json_tree(value, ser_ctx)
    assert json.dumps(tree1, sort_keys=True) == json.dumps(tree2, sort_keys=True)


def test_bytes_round_trip(ctx_pair):
    value = b"\x00\x01\xff hello"
    result = round_trip(value, ctx_pair)
    assert result == value
    assert isinstance(result, bytes)


def test_datetime_round_trip(ctx_pair):
    value = datetime.datetime(2024, 3, 14, 15, 9, 26)
    result = round_trip(value, ctx_pair)
    assert result == value
    assert isinstance(result, datetime.datetime)


def test_date_round_trip(ctx_pair):
    value = datetime.date(2024, 3, 14)
    result = round_trip(value, ctx_pair)
    assert result == value
    assert isinstance(result, datetime.date)
    assert not isinstance(result, datetime.datetime)


def test_non_string_dict_key_rejected(ctx_pair):
    ser_ctx, _ = ctx_pair
    with pytest.raises(SerializationError, match="dict keys must be strings"):
        to_json_tree({1: "a"}, ser_ctx)


def test_reserved_type_key_rejected(ctx_pair):
    ser_ctx, _ = ctx_pair
    with pytest.raises(SerializationError, match="__type__"):
        to_json_tree({"__type__": "whatever"}, ser_ctx)


def test_deeply_nested_arg_like_structure_round_trip(ctx_pair):
    value = {
        "args": [1, "two", [3, 4], {"nested": True}],
        "kwargs": {"scale": 2.0, "flags": (True, False)},
    }
    assert round_trip(value, ctx_pair) == value
