import json

import numpy as np
import pandas as pd
import pytest

from regression_records.exceptions import DeserializationError, SerializationError
from regression_records.serialization import (
    DeserializationContext,
    SerializationContext,
    default_registry,
    from_json_tree,
    to_json_tree,
)
from regression_records.serialization.pandas_serializer import (
    PandasDataFrameSerializer,
    PandasSeriesSerializer,
)

from ._helpers import round_trip


def test_dataframe_round_trip(ctx_pair):
    df = pd.DataFrame(
        {"int": [1, 2, 3], "float": [1.0, 2.5, 3.5], "str": ["a", "b", "c"]}
    )
    result = round_trip(df, ctx_pair)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_datetime_index(ctx_pair):
    df = pd.DataFrame(
        {"x": [1, 2, 3]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    result = round_trip(df, ctx_pair)
    # parquet does not preserve pandas' DatetimeIndex.freq metadata - a known,
    # documented limitation of parquet-based storage, not a bug in the serializer.
    pd.testing.assert_frame_equal(result, df, check_freq=False)


def test_series_round_trip(ctx_pair):
    s = pd.Series([1, 2, 3], name="values")
    result = round_trip(s, ctx_pair)
    pd.testing.assert_series_equal(result, s)


def test_series_without_name_round_trip(ctx_pair):
    s = pd.Series([1.0, 2.0, None])
    result = round_trip(s, ctx_pair)
    pd.testing.assert_series_equal(result, s)


MIXED = pd.DataFrame({"mixed": [1, "two", 3.0, None]})


@pytest.fixture
def pickling_ctx_pair(tmp_path):
    """A context pair whose pandas serializers have opted in to pickle."""
    registry = default_registry.copy()
    registry.register(PandasDataFrameSerializer(allow_pickle=True), replace=True)
    registry.register(PandasSeriesSerializer(allow_pickle=True), replace=True)
    return (
        SerializationContext(data_dir=tmp_path, registry=registry),
        DeserializationContext(base_dir=tmp_path, registry=registry),
    )


def test_a_frame_parquet_cannot_hold_is_refused_by_default(ctx_pair, tmp_path):
    with pytest.raises(SerializationError, match="allow_pickle=True"):
        round_trip(MIXED, ctx_pair)
    assert list(tmp_path.iterdir()) == []


def test_pickle_fallback_is_available_by_opting_in(pickling_ctx_pair):
    with pytest.warns(UserWarning, match="falling back to pickle"):
        result = round_trip(MIXED, pickling_ctx_pair)
    pd.testing.assert_frame_equal(result, MIXED)


def test_a_record_cannot_ask_for_a_file_to_be_unpickled(
    ctx_pair, pickling_ctx_pair, tmp_path
):
    """``format`` is a field of the record, so editing a record must not be enough."""
    with pytest.warns(UserWarning):
        tree = to_json_tree(MIXED, pickling_ctx_pair[0])
    assert tree["format"] == "pickle"

    with pytest.raises(DeserializationError, match="unpickling runs code"):
        from_json_tree(tree, ctx_pair[1])


def test_a_missing_parquet_engine_is_an_error_not_a_reason_to_pickle(
    pickling_ctx_pair, monkeypatch
):
    """Without pyarrow every frame fails, so falling back would pickle the whole store."""

    def no_engine(self, *args, **kwargs):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow'")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
    with pytest.raises(SerializationError, match=r"regression-records\[pandas\]"):
        to_json_tree(pd.DataFrame({"a": [1]}), pickling_ctx_pair[0])


@pytest.mark.parametrize("name", [3, np.int64(3), ("level", 1)])
def test_series_with_a_non_string_name_round_trips(ctx_pair, name):
    """Parquet needs string column names; ``json.dump`` needs JSON-safe payloads."""
    series = pd.Series([1.0, 2.0], name=name)
    tree = to_json_tree(series, ctx_pair[0])
    json.dumps(tree)
    pd.testing.assert_series_equal(from_json_tree(tree, ctx_pair[1]), series)


def test_nested_dataframe_in_dict(ctx_pair):
    value = {"df": pd.DataFrame({"a": [1, 2]}), "meta": {"rows": 2}}
    result = round_trip(value, ctx_pair)
    pd.testing.assert_frame_equal(result["df"], value["df"])
    assert result["meta"] == {"rows": 2}


def test_a_categorical_column_keeps_its_categories_and_order(ctx_pair):
    frame = pd.DataFrame(
        {"c": pd.Categorical(["a", "b"], categories=["b", "a", "z"], ordered=True)}
    )

    back = round_trip(frame, ctx_pair)

    assert back["c"].cat.categories.tolist() == ["b", "a", "z"]
    assert back["c"].cat.ordered


@pytest.mark.parametrize("as_series", [False, True], ids=["frame", "series"])
def test_an_object_string_column_validates_against_itself(ctx_pair, as_series):
    # pandas 3 reads any string column back from parquet as ``str``.
    value = pd.Series(["a", None, "c"], dtype=object, name="s")
    if not as_series:
        value = value.to_frame()

    from regression_records.comparison import compare

    assert compare(value, round_trip(value, ctx_pair)).passed
