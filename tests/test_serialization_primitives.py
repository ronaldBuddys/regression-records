import datetime
import json
from typing import NamedTuple

import pytest

from regression_records.exceptions import IdentityResolutionError, SerializationError
from regression_records.serialization import from_json_tree, to_json_tree
from regression_records.serialization.base import (
    DeserializationContext,
    SerializationContext,
    SerializerRegistry,
)
from regression_records.serialization.object_serializer import GenericObjectSerializer
from regression_records.serialization.primitives import DatetimeSerializer

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


# ------------------------------------------------------------------ registry dispatch


class ReversedTupleSerializer:
    """A stand-in third party claiming a type tag the library also ships."""

    type_tag = "tuple"

    def can_handle(self, obj):
        return isinstance(obj, tuple)

    def store(self, obj, ctx):
        return {"reversed_items": list(reversed(obj))}

    def load(self, blob, ctx):
        return tuple(reversed(blob["reversed_items"]))


def test_a_registered_serializer_can_take_over_a_builtin_tag(tmp_path):
    """Loading must dispatch through the registry, not through a hard-coded tag list.

    tuple/set/bytes were once written by the walker and read back by an if-chain beside
    it, which meant a serializer registered for one of those tags was honoured on the
    way out and silently ignored on the way in - producing a value that didn't round
    trip.
    """
    registry = SerializerRegistry()
    registry.register(ReversedTupleSerializer())
    ser_ctx = SerializationContext(data_dir=tmp_path, registry=registry)
    deser_ctx = DeserializationContext(base_dir=tmp_path, registry=registry)

    tree = to_json_tree((1, 2, 3), ser_ctx)
    assert tree["reversed_items"] == [3, 2, 1]
    assert from_json_tree(tree, deser_ctx) == (1, 2, 3)


def test_registering_a_serializer_beats_the_generic_fallback_but_not_the_specific_ones():
    registry = SerializerRegistry()
    specific, custom, fallback = (
        DatetimeSerializer(),
        ReversedTupleSerializer(),
        GenericObjectSerializer(),
    )
    registry.register(specific)
    registry.register(fallback, fallback=True)
    registry.register(custom)

    assert registry.find((1, 2)) is custom
    assert registry.find(datetime.datetime(2024, 1, 1)) is specific
    assert registry.find(_Plain()) is fallback


def test_a_second_fallback_still_sits_behind_ordinary_serializers():
    class SecondFallback(GenericObjectSerializer):
        type_tag = "object.second"  # a tag can only be registered once

    registry = SerializerRegistry()
    first, second, custom = (GenericObjectSerializer(), SecondFallback(), _AnyObject())
    registry.register(first, fallback=True)
    registry.register(second, fallback=True)
    registry.register(custom)

    assert registry.find(_Plain()) is custom


def test_copy_preserves_dispatch_order_and_tags():
    registry = SerializerRegistry()
    specific, fallback = DatetimeSerializer(), GenericObjectSerializer()
    registry.register(specific)
    registry.register(fallback, fallback=True)

    clone = registry.copy()

    assert len(clone) == len(registry)
    assert clone.find(datetime.datetime(2024, 1, 1)) is specific
    assert clone.find(_Plain()) is fallback
    assert clone.by_tag("datetime.datetime") is specific


def test_registering_on_a_copy_leaves_the_original_alone():
    """What makes ``serializer_registry=`` a real alternative to global registration."""
    registry = SerializerRegistry()
    registry.register(GenericObjectSerializer(), fallback=True)

    clone = registry.copy()
    custom = ReversedTupleSerializer()
    clone.register(custom)

    assert clone.find((1, 2)) is custom
    assert registry.find((1, 2)) is not custom
    assert len(registry) == 1


class _Plain:
    def __init__(self):
        self.x = 1


class _AnyObject:
    type_tag = "any-object"

    def can_handle(self, obj):
        return isinstance(obj, _Plain)

    def store(self, obj, ctx):  # pragma: no cover - ordering test only
        return {}

    def load(self, blob, ctx):  # pragma: no cover - ordering test only
        return _Plain()


def test_a_type_tag_can_only_be_registered_once():
    """Values are written by first match but read by tag, so a clash splits the two."""
    registry = SerializerRegistry()
    registry.register(DatetimeSerializer())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(DatetimeSerializer())


def test_replace_swaps_a_serializer_in_place():
    registry = SerializerRegistry()
    first, other, replacement = (
        DatetimeSerializer(),
        GenericObjectSerializer(),
        DatetimeSerializer(),
    )
    registry.register(first)
    registry.register(other, fallback=True)

    registry.register(replacement, replace=True)

    assert len(registry) == 2
    assert registry.find(datetime.datetime(2024, 1, 1)) is replacement
    assert registry.by_tag("datetime.datetime") is replacement
    assert registry.find(_Plain()) is other


class Reading(NamedTuple):
    """Module-level, because a recorded named tuple is re-imported by name."""

    label: str
    value: float


def test_named_tuple_round_trips_as_itself(ctx_pair):
    """Stored as a plain tuple it could be written but never validated."""
    result = round_trip(Reading("spot", 101.5), ctx_pair)
    assert result == Reading("spot", 101.5)
    assert type(result) is Reading


def test_a_record_cannot_make_a_named_tuple_load_call_a_constructor(ctx_pair):
    """The class is named by the record, so it is never called - only tuple.__new__."""
    tree = to_json_tree(Reading("spot", 1.0), ctx_pair[0])
    tree["class"] = {"module": "subprocess", "qualname": "Popen"}
    with pytest.raises(IdentityResolutionError, match="not a named tuple"):
        from_json_tree(tree, ctx_pair[1])


def test_a_self_referencing_container_is_refused_not_recursed_into(ctx_pair):
    cyclic: list = [1]
    cyclic.append(cyclic)
    with pytest.raises(SerializationError, match="reference back to itself"):
        to_json_tree(cyclic, ctx_pair[0])


def test_an_object_referenced_twice_is_not_a_cycle(ctx_pair):
    shared = [1, 2]
    assert round_trip({"a": shared, "b": shared}, ctx_pair) == {
        "a": [1, 2],
        "b": [1, 2],
    }


def test_a_serializer_payload_cannot_replace_its_type_tag(tmp_path):
    class Impostor:
        type_tag = "impostor"

        def can_handle(self, obj):
            return isinstance(obj, complex)

        def store(self, obj, ctx):
            return {"__type__": "bytes", "hex": "00"}

        def load(self, blob, ctx):
            return 0j

    registry = SerializerRegistry()
    registry.register(Impostor())
    ctx = SerializationContext(data_dir=tmp_path, registry=registry)
    with pytest.raises(SerializationError, match="reserved key"):
        to_json_tree(1j, ctx)
