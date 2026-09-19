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
    RecordNotFoundError,
    RegressionMismatchError,
    RegressionRecordsError,
    SerializationError,
    UnmatchedInvocationError,
    record_func,
)
from regression_records.comparison import ComparisonResult, Diff
from regression_records.config import OnNewCase, RunMode
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


# ------------------------------------------------------------------------- hierarchy


@pytest.mark.parametrize(
    "exc_type",
    [
        SerializationError,
        DeserializationError,
        CorruptRecordError,
        RecordNotFoundError,
        UnmatchedInvocationError,
        RegressionMismatchError,
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


def test_unserializable_value_points_at_writing_a_serializer(store):
    def emit():
        return lambda x: x  # a function has a __dict__, but an empty one

    with pytest.raises(SerializationError, match="custom Serializer"):
        record_func(store)(emit)()


def test_a_registry_with_no_fallback_names_the_registration_function(tmp_path):
    """The other unserializable path: nothing claimed the value at all.

    Unreachable with the default registry - since Python 3.11 every object has an
    inherited ``__getstate__``, so the generic fallback claims everything and reports
    the message above instead. It is live for a caller who injects a narrow registry.
    """
    from regression_records.serialization import SerializationContext, to_json_tree
    from regression_records.serialization.base import SerializerRegistry

    ctx = SerializationContext(data_dir=tmp_path, registry=SerializerRegistry())
    with pytest.raises(SerializationError, match="register_serializer"):
        to_json_tree(object(), ctx)


def test_object_dtype_array_explains_why_it_is_refused(store):
    def emit():
        return np.array([{"a": 1}, "b"], dtype=object)

    with pytest.raises(SerializationError, match="dtype=object"):
        record_func(store)(emit)()


def test_non_string_dict_key_suggests_the_workaround(store):
    def emit():
        return {1: "one"}

    with pytest.raises(SerializationError, match="convert keys to strings"):
        record_func(store)(emit)()


def test_class_defined_in_main_is_refused_with_the_reason(store, monkeypatch):
    class Local:
        def __init__(self):
            self.x = 1

    monkeypatch.setattr(Local, "__module__", "__main__")

    def emit():
        return Local()

    with pytest.raises(SerializationError, match="__main__"):
        record_func(store)(emit)()


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
    def echo(arr):
        return arr

    recorded = record_func(store)(echo)
    recorded(np.arange(4))
    for stale in store._data_dir(recorded.identity).glob("*.npy"):
        stale.unlink()

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(echo)
    with pytest.raises(CorruptRecordError, match="missing"):
        validating(np.arange(4))


# ------------------------------------------------------------ IdentityResolutionError


def test_unimportable_recorded_class_names_module_and_class(store):
    recorded = record_func(store, name="emit_payload")(_emit_payload)
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


# --------------------------------------------------- lookup and matching failures


def test_record_not_found_tells_you_to_record_first(store):
    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(add)
    with pytest.raises(RecordNotFoundError, match="record a run first"):
        validating(1, 2)


def test_unmatched_invocation_reports_the_input_hash(store):
    record_func(store)(add)(1, 2)
    strict = record_func(store, on_new_case=OnNewCase.ERROR)(add)
    with pytest.raises(UnmatchedInvocationError, match="input_hash="):
        strict(9, 9)


# ------------------------------------------------------------- RegressionMismatchError


def test_mismatch_error_carries_the_result_and_summarizes_it(store):
    def compute(x):
        return x * 2

    record_func(store)(compute)(5)

    def compute(x):  # noqa: F811
        return x * 3

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(compute)
    with pytest.raises(RegressionMismatchError) as excinfo:
        validating(5)

    error = excinfo.value
    assert isinstance(error.result, ComparisonResult)
    assert not error.result.passed
    # The message is the result's own summary, so the diff paths reach a bare traceback.
    assert "$.output" in str(error)
    assert error.result.summary() == str(error)


def test_mismatch_error_can_be_built_from_a_result_directly():
    result = ComparisonResult.from_diffs(
        [Diff(path="$.output", kind="value_mismatch", expected="1", actual="2", message="differs")]
    )
    assert "differs" in str(RegressionMismatchError(result))
