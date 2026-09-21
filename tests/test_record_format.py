"""The record format: schema version 2, and reading version 1 records unchanged."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from regression_records.comparison import (
    compare,
    compare_exceptions,
    exception_type_name,
)
from regression_records.exceptions import SerializationError
from regression_records.identity import FunctionIdentity
from regression_records.records import (
    SCHEMA_VERSION,
    Invocation,
    RegressionRecord,
    SerializedValue,
)
from regression_records.serialization import to_json_tree
from regression_records.store import RecordStore

V1_STORE = Path(__file__).parent / "fixtures" / "store_v1"


def _v1_identity(qualname: str) -> FunctionIdentity:
    # Only module and qualname locate records; the rest is informational.
    return FunctionIdentity(
        module="v1_fixture_source",
        qualname=qualname,
        kind="function",
        code_hash="",
        signature="",
    )


def _load_v1(qualname: str) -> tuple[RegressionRecord, RecordStore]:
    store = RecordStore(V1_STORE)
    [record] = store.list_records(_v1_identity(qualname))
    assert record.schema.schema_version == 1
    return record, store


def _strict_json(text: str):
    """Parse ``text`` as RFC 8259 JSON, which has no NaN or Infinity literal."""

    def reject(token):
        raise ValueError(f"non-standard JSON token {token!r}")

    return json.loads(text, parse_constant=reject)


# --- version 1 records ------------------------------------------------------------


def test_the_v1_fixture_really_holds_bare_non_finite_tokens():
    text = next((V1_STORE / "v1_fixture_source" / "non_finite").rglob("*.json"))
    with pytest.raises(ValueError, match="non-standard"):
        _strict_json(text.read_text())


def test_v1_python_floats_load_and_compare_equal_to_a_fresh_result():
    record, store = _load_v1("non_finite")
    output = record.load_output(store.deserialization_context(record.identity))

    assert math.isnan(output["nan"])
    assert output["inf"] == math.inf and output["ninf"] == -math.inf
    fresh = {"nan": math.nan, "inf": math.inf, "ninf": -math.inf, "x": 1.5}
    assert compare(output, fresh).passed


def test_v1_numpy_scalars_load_with_their_dtypes():
    record, store = _load_v1("numpy_non_finite")
    output = record.load_output(store.deserialization_context(record.identity))

    assert [type(v) for v in output[:3]] == [np.float64, np.float64, np.float32]
    fresh = [np.float64("nan"), np.float64("inf"), np.float32("-inf"), 2.5]
    assert compare(output, fresh).passed


def test_v1_bare_exception_names_compare_by_bare_name():
    record, _ = _load_v1("raises")

    assert record.exception.exc_type == "ValueError"
    assert compare_exceptions(record.exception.exc_type, ValueError("x")).passed
    assert not compare_exceptions(record.exception.exc_type, KeyError("x")).passed


# --- exception names ------------------------------------------------------------------


class ConnectionError(Exception):
    """Same bare name as the builtin, from a different module."""


class Outer:
    class Inner(Exception):
        pass


def test_exception_names_are_module_qualified():
    assert exception_type_name(ValueError()) == "builtins.ValueError"
    assert exception_type_name(ConnectionError()) == f"{__name__}.ConnectionError"
    assert exception_type_name(Outer.Inner()) == f"{__name__}.Outer.Inner"


def test_same_named_exceptions_from_different_modules_differ():
    import builtins

    recorded = exception_type_name(builtins.ConnectionError())
    result = compare_exceptions(recorded, ConnectionError())

    assert not result.passed
    assert result.diffs[0].expected == "builtins.ConnectionError"
    assert result.diffs[0].actual == f"{__name__}.ConnectionError"
    assert compare_exceptions(recorded, builtins.ConnectionError()).passed


def test_a_v1_bare_name_cannot_tell_same_named_exceptions_apart():
    # The documented limit of version 1 records.
    assert compare_exceptions("ConnectionError", ConnectionError()).passed


def test_a_nested_exception_class_matches_its_qualified_name():
    assert compare_exceptions(f"{__name__}.Outer.Inner", Outer.Inner()).passed


def test_returning_where_a_qualified_exception_was_recorded_is_a_mismatch():
    result = compare_exceptions("builtins.ValueError", None)

    assert not result.passed
    assert "builtins.ValueError" in result.diffs[0].message


# --- non-finite floats ----------------------------------------------------------------

NON_FINITE = ["nan", "inf", "-inf"]


def sample_function(x):
    return x


def _save(store: RecordStore, arg, output) -> Path:
    identity = FunctionIdentity.from_function(sample_function)
    ctx = store.serialization_context(identity)
    record = RegressionRecord.new(
        identity=identity,
        input_hash="h",
        invocation=Invocation(
            args=SerializedValue(to_json_tree([arg], ctx)),
            kwargs=SerializedValue({}),
        ),
        output=SerializedValue(to_json_tree(output, ctx)),
    )
    return store.save(record)


@pytest.mark.parametrize("text", NON_FINITE)
@pytest.mark.parametrize(
    "make", [float, np.float64, np.float32], ids=lambda f: f.__name__
)
def test_non_finite_floats_are_written_as_strict_json(tmp_path, make, text):
    store = RecordStore(tmp_path)
    value = make(text)
    path = _save(store, arg=value, output={"v": value, "items": [value]})

    raw = _strict_json(path.read_text(encoding="utf-8"))
    assert raw["schema"]["schema_version"] == SCHEMA_VERSION == 2

    identity = FunctionIdentity.from_function(sample_function)
    [record] = store.list_records(identity)
    ctx = store.deserialization_context(identity)
    output = record.load_output(ctx)
    assert type(output["v"]) is type(value)
    assert compare(output, {"v": value, "items": [value]}).passed
    assert compare(record.load_args(ctx), [value]).passed


def test_a_python_nan_is_tagged_in_the_tree(tmp_path):
    store = RecordStore(tmp_path)
    ctx = store.serialization_context(FunctionIdentity.from_function(sample_function))

    assert to_json_tree(math.nan, ctx) == {"__type__": "float", "value": "nan"}
    assert to_json_tree(-math.inf, ctx) == {"__type__": "float", "value": "-inf"}
    assert to_json_tree(1.5, ctx) == 1.5


class _LeakyNan:
    """A value whose custom serializer puts a raw NaN in its payload."""


class _LeakySerializer:
    type_tag = "test.leaky"

    def can_handle(self, obj):
        return isinstance(obj, _LeakyNan)

    def store(self, obj, ctx):
        return {"value": math.nan}

    def load(self, blob, ctx):
        return _LeakyNan()


def test_a_raw_nan_from_a_custom_serializer_is_refused_on_save(tmp_path):
    store = RecordStore(tmp_path)
    store.registry = store.registry.copy()
    store.registry.register(_LeakySerializer())

    with pytest.raises(SerializationError, match="strict JSON"):
        _save(store, arg=1, output=_LeakyNan())
    assert not store.has_records(FunctionIdentity.from_function(sample_function))


def test_renaming_keeps_a_v1_record_with_bare_nan_writable(tmp_path):
    shutil.copytree(V1_STORE, tmp_path, dirs_exist_ok=True)
    store = RecordStore(tmp_path)
    new = _v1_identity("renamed")

    store.rename_identity(_v1_identity("non_finite"), new)

    [record] = store.list_records(new)
    assert record.schema.schema_version == 1
    output = record.load_output(store.deserialization_context(new))
    assert math.isnan(output["nan"])
