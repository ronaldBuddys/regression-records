"""Objects stored, compared and rebuilt by their attributes."""

import dataclasses
import io
import threading
import types

import numpy as np
import pytest

from regression_records import compare
from regression_records._attributes import attributes, stored_by_attributes
from regression_records.exceptions import DeserializationError, SerializationError
from regression_records.serialization import from_json_tree, to_json_tree

from ._helpers import round_trip
from ._models import Grid, Holder, Labels, Quote, Vec, spx_like_universe


def paths(result):
    return [d.path for d in result.diffs]


# --- round trips ------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda u: u.surface._labels,
        lambda u: u.curves["USD"],
        lambda u: u.surface,
        lambda u: u,
        lambda u: Vec(1.5, "s"),
        lambda u: Quote(99.5, 100.5),
        lambda u: types.MappingProxyType({"a": np.arange(3), "b": Quote(1.0, 2.0)}),
    ],
    ids=[
        "slots-mapping",
        "dict-class",
        "subclass",
        "frozen-slots-dataclass",
        "slots-class",
        "namedtuple",
        "mappingproxy",
    ],
)
def test_round_trips_and_compares_equal_by_attributes(build, ctx_pair):
    original = build(spx_like_universe())

    loaded = round_trip(original, ctx_pair)

    assert type(loaded) is type(original)
    result = compare(original, loaded)
    assert result.passed, result.summary()


def test_a_frozen_slotted_dataclass_is_rebuilt_without_post_init(ctx_pair):
    universe = spx_like_universe()
    tree = to_json_tree(universe, ctx_pair[0])
    # An empty snap list fails __post_init__'s check; loading must not run it.
    tree["attrs"]["snap_days"] = to_json_tree(np.array([]), ctx_pair[0])

    loaded = from_json_tree(tree, ctx_pair[1])

    assert len(loaded.snap_days) == 0
    assert isinstance(loaded.curves, types.MappingProxyType)
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.symbol = "NDX"


def test_lookup_caches_keyed_by_numpy_scalars_round_trip(ctx_pair):
    labels = spx_like_universe().surface._labels

    loaded = round_trip(labels, ctx_pair)

    assert loaded._lookups == labels._lookups
    key = next(iter(loaded._lookups["trade_date"]))
    assert isinstance(key, np.datetime64)
    assert loaded.position("strike", np.int64(110)) == 2


def test_a_slots_class_keeps_mangled_slots_and_leaves_unset_ones_unset(ctx_pair):
    loaded = round_trip(Vec(1.5, "s"), ctx_pair)

    assert loaded.x == 1.5
    assert loaded.secret == "s"
    assert not hasattr(loaded, "maybe")


def test_a_named_tuple_is_stored_by_field_name(ctx_pair):
    tree = to_json_tree(Quote(99.5, 100.5), ctx_pair[0])

    assert tree["__type__"] == "object"
    assert tree["attrs"] == {"bid": 99.5, "ask": 100.5}
    assert round_trip(Quote(99.5, 100.5), ctx_pair).mid == 100.0


def test_class_attributes_and_properties_are_not_stored():
    assert set(attributes(spx_like_universe().surface)) == {
        "_backend",
        "_data",
        "_labels",
        "_attrs",
    }
    assert set(attributes(Quote(1.0, 2.0))) == {"bid", "ask"}


# --- comparison paths -------------------------------------------------------------


def test_named_tuple_diffs_name_the_field():
    result = compare(Quote(99.5, 100.5), Quote(99.5, 101.0), root="$.output")
    assert paths(result) == ["$.output.ask"]


def test_slot_and_nested_dataclass_diffs_name_the_attribute():
    expected, actual = spx_like_universe(), spx_like_universe()
    actual.surface._data["implied_vol"] = actual.surface._data["implied_vol"] + 1.0

    assert paths(compare(Vec(1.0, "s"), Vec(2.0, "s"))) == ["$.x"]
    assert paths(compare(expected, actual)) == ["$.surface._data.implied_vol"]


def test_objects_are_compared_by_attributes_not_their_loose_eq():
    assert Labels(a=[1]) == Labels(a=[2])
    assert paths(compare(Labels(a=[1]), Labels(a=[2]))) == ["$._labels.a"]


def test_a_proxy_and_a_dict_are_different_types():
    result = compare(types.MappingProxyType({"a": 1}), {"a": 1})
    assert [d.kind for d in result.diffs] == ["type_mismatch"]


# --- refusals ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "held",
    [len, lambda x: x, threading.Lock(), Grid],
    ids=["builtin-function", "lambda", "lock", "class"],
)
def test_an_attribute_that_cannot_be_rebuilt_refuses_the_whole_value(held, ctx_pair):
    with pytest.raises(SerializationError, match="cannot be rebuilt"):
        to_json_tree(Holder(held), ctx_pair[0])


def test_an_open_file_is_refused(ctx_pair, tmp_path):
    with (
        (tmp_path / "f.txt").open("w") as handle,
        pytest.raises(SerializationError, match="'_io.TextIOWrapper'"),
    ):
        to_json_tree(Holder(handle), ctx_pair[0])


def test_an_object_referring_back_to_itself_is_refused(ctx_pair):
    holder = Holder(None)
    holder.held = holder
    with pytest.raises(SerializationError, match="reference back to itself"):
        to_json_tree(holder, ctx_pair[0])


@pytest.mark.parametrize(
    ("value", "stored"),
    [
        (Vec(1.0, "s"), True),
        (Quote(1.0, 2.0), True),
        (threading.Lock(), False),
        (np.float64(1.0), False),
        (Grid, False),
        (print, False),
        # C classes holding state below their attributes, though object.__new__
        # accepts them: a buffered text stream, an exception, a namespace.
        (io.StringIO(), False),
        (ValueError("x"), False),
        (types.SimpleNamespace(a=1), False),
    ],
)
def test_which_values_are_stored_by_attributes(value, stored):
    assert stored_by_attributes(value) is stored


# --- attributes that no longer fit the class --------------------------------------


def test_a_removed_slot_is_a_clear_load_error(ctx_pair):
    tree = to_json_tree(Vec(1.0, "s"), ctx_pair[0])
    tree["attrs"]["gone"] = 1
    with pytest.raises(DeserializationError, match="'gone' no longer exists"):
        from_json_tree(tree, ctx_pair[1])


def test_a_dataclass_field_added_since_recording_is_a_clear_load_error(ctx_pair):
    tree = to_json_tree(spx_like_universe(), ctx_pair[0])
    del tree["attrs"]["symbol"]
    with pytest.raises(DeserializationError, match=r"field\(s\) \['symbol'\]"):
        from_json_tree(tree, ctx_pair[1])


def test_a_changed_named_tuple_is_a_clear_load_error(ctx_pair):
    tree = to_json_tree(Quote(1.0, 2.0), ctx_pair[0])
    tree["attrs"]["last"] = tree["attrs"].pop("ask")
    with pytest.raises(
        DeserializationError, match=r"added \['ask'\], removed \['last'\]"
    ):
        from_json_tree(tree, ctx_pair[1])


# --- records written before 0.4.0 -------------------------------------------------


def test_a_schema_2_object_loads_from_its_state(ctx_pair):
    tree = {
        "__type__": "object",
        "class": {"module": Vec.__module__, "qualname": "Vec"},
        "state": {"x": 1.0, "_Vec__secret": "s"},
    }
    assert compare(Vec(1.0, "s"), from_json_tree(tree, ctx_pair[1]))


def test_a_schema_2_named_tuple_loads_from_its_items(ctx_pair):
    tree = {
        "__type__": "namedtuple",
        "class": {"module": Quote.__module__, "qualname": "Quote"},
        "items": [1.0, 2.0],
    }
    loaded = from_json_tree(tree, ctx_pair[1])
    assert type(loaded) is Quote
    assert loaded == Quote(1.0, 2.0)
