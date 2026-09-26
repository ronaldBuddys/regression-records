"""Each exception type in the scenario that raises it, and what its message must say.

The decorator and serialization suites reach most of these incidentally. This file is
about the messages themselves: an exception a user hits at 2am is a diagnostic tool, and
a message that doesn't name the escape hatch is a bug in its own right.
"""

import json

import numpy as np
import pytest

from regression_records import (
    CorruptRecordError,
    DeserializationError,
    IdentityResolutionError,
    RegressionRecordsError,
    SerializationError,
    record_func,
)
from regression_records.serialization import (
    DeserializationContext,
    default_registry,
    from_json_tree,
)
from regression_records.store import RecordStore


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path)


def add(a, b=1):
    return a + b


def echo(arr):
    return arr


# ------------------------------------------------------------------------- hierarchy


@pytest.mark.parametrize(
    "exc_type",
    [
        SerializationError,
        DeserializationError,
        CorruptRecordError,
        IdentityResolutionError,
    ],
)
def test_every_exception_is_catchable_as_the_base_type(exc_type):
    """One ``except RegressionRecordsError`` must catch everything this library raises."""
    assert issubclass(exc_type, RegressionRecordsError)


def test_corrupt_record_is_a_deserialization_error():
    # Lets a caller treat "the store is damaged" as one case without enumerating types.
    assert issubclass(CorruptRecordError, DeserializationError)


# -------------------------------------------------------------------- SerializationError


def test_unserializable_value_points_at_a_leaf_serializer(store):
    # A function is code, not data: it has no attributes to rebuild it from.
    with pytest.raises(SerializationError, match="register a leaf Serializer"):
        record_func(store)(echo)(lambda x: x)


def test_object_dtype_array_explains_why_it_is_refused(store):
    with pytest.raises(SerializationError, match="dtype=object"):
        record_func(store)(echo)(np.array([{"a": 1}, "b"], dtype=object))


def test_a_non_string_dict_key_records(store):
    assert record_func(store)(echo)({1: "one"}) == {1: "one"}


def test_class_defined_in_main_is_refused_with_the_reason(store, monkeypatch):
    class Local:
        def __init__(self):
            self.x = 1

    monkeypatch.setattr(Local, "__module__", "__main__")

    with pytest.raises(SerializationError, match="__main__"):
        record_func(store)(echo)(Local())


# ------------------------------------------------------ DeserializationError / corruption


def test_unknown_type_tag_blames_the_library_version(tmp_path):
    ctx = DeserializationContext(base_dir=tmp_path, registry=default_registry)
    with pytest.raises(DeserializationError, match="different serializer support"):
        from_json_tree({"__type__": "some.future.Type", "value": 1}, ctx)


def test_malformed_record_json_names_the_file(store):
    recorded = record_func(store)(add)
    recorded(1, 2)
    path = next(store._records_dir(recorded.identity).glob("*.json"))
    path.write_text("{not json at all")

    with pytest.raises(CorruptRecordError, match=str(path.name)):
        store.list_records(recorded.identity)


def test_record_missing_a_required_field_is_corrupt(store):
    recorded = record_func(store)(add)
    recorded(1, 2)
    path = next(store._records_dir(recorded.identity).glob("*.json"))
    payload = json.loads(path.read_text())
    del payload["identity"]
    path.write_text(json.dumps(payload))

    with pytest.raises(CorruptRecordError, match="malformed"):
        store.list_records(recorded.identity)


def test_missing_side_file_is_reported_as_corruption(store):
    recorded = record_func(store)(echo)
    recorded(np.arange(4))
    for stale in store._data_dir(recorded.identity).glob("*.npy"):
        stale.unlink()

    (record,) = store.list_records(recorded.identity)
    with pytest.raises(CorruptRecordError, match="missing"):
        record.load_args(store.deserialization_context(recorded.identity))


# ------------------------------------------------------------ IdentityResolutionError


def test_unimportable_recorded_class_names_module_and_class(store):
    recorded = record_func(store)(_emit_payload)
    recorded()

    path = next(store._records_dir(recorded.identity).glob("*.json"))
    payload = json.loads(path.read_text())
    payload["output"]["class"]["qualname"] = "_VanishedPayload"
    path.write_text(json.dumps(payload))

    record = store.list_records(recorded.identity)[0]
    ctx = store.deserialization_context(recorded.identity)
    with pytest.raises(IdentityResolutionError, match="_VanishedPayload"):
        record.load_output(ctx)


class _Payload:
    def __init__(self):
        self.value = 1


def _emit_payload():
    return _Payload()
