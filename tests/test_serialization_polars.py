import datetime

import polars as pl
import pytest

from regression_records.exceptions import CorruptRecordError, SerializationError
from regression_records.serialization import from_json_tree, to_json_tree

from ._helpers import round_trip


def test_dataframe_round_trip(ctx_pair):
    df = pl.DataFrame(
        {
            "int": [1, 2, 3],
            "float": [1.0, 2.5, None],
            "str": ["a", "b", None],
            "bool": [True, False, True],
        }
    )
    result = round_trip(df, ctx_pair)
    assert result.equals(df)


def test_dataframe_datetime_and_temporal_types(ctx_pair):
    df = pl.DataFrame(
        {
            "dt": [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)],
            "date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
            "dur": [datetime.timedelta(days=1), datetime.timedelta(hours=3)],
        }
    )
    result = round_trip(df, ctx_pair)
    assert result.equals(df)
    assert result.schema == df.schema


def test_dataframe_nested_list_and_struct_columns(ctx_pair):
    df = pl.DataFrame(
        {
            "tags": [["a", "b"], ["c"], []],
            "info": [{"x": 1, "y": "p"}, {"x": 2, "y": "q"}, {"x": 3, "y": "r"}],
        }
    )
    result = round_trip(df, ctx_pair)
    assert result.equals(df)
    assert result.schema == df.schema


def test_empty_dataframe_round_trip(ctx_pair):
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    result = round_trip(df, ctx_pair)
    assert result.equals(df)
    assert result.schema == df.schema


def test_categorical_column_round_trip(ctx_pair):
    df = pl.DataFrame({"c": pl.Series(["x", "y", "x"], dtype=pl.Categorical)})
    result = round_trip(df, ctx_pair)
    assert result["c"].cast(pl.Utf8).to_list() == df["c"].cast(pl.Utf8).to_list()


def test_content_hash_dedup(ctx_pair, tmp_path):
    ser_ctx, _ = ctx_pair
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"a": [1, 2, 3]})
    tree1 = to_json_tree(df1, ser_ctx)
    tree2 = to_json_tree(df2, ser_ctx)
    assert tree1["path"] == tree2["path"]
    assert len(list(tmp_path.glob("*.parquet"))) == 1


def test_lazyframe_store_rejected(ctx_pair):
    ser_ctx, _ = ctx_pair
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    with pytest.raises(SerializationError, match="collect"):
        to_json_tree(lf, ser_ctx)


@pytest.mark.parametrize("as_series", [False, True], ids=["frame", "series"])
def test_object_dtype_is_refused_naming_the_column(ctx_pair, tmp_path, as_series):
    ser_ctx, _ = ctx_pair
    payloads = pl.Series("payload", [object(), object()], dtype=pl.Object)
    value = payloads if as_series else pl.DataFrame({"id": [1, 2], "payload": payloads})
    with pytest.raises(SerializationError, match=r"\['payload'\].*Object"):
        to_json_tree(value, ser_ctx)
    assert not list(tmp_path.iterdir())


def test_nested_dataframe_in_dict(ctx_pair):
    value = {"df": pl.DataFrame({"a": [1, 2]}), "meta": {"rows": 2}}
    result = round_trip(value, ctx_pair)
    assert result["df"].equals(value["df"])
    assert result["meta"] == {"rows": 2}


# ---------------------------------------------------------------------------- series


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("price", [1.0, None, 3.5]),
        pl.Series([1, 2, 3]),
        pl.Series("empty", [], dtype=pl.Float64),
        pl.Series("nothing", [None, None]),
        pl.Series("tags", [["a"], [], None]),
    ],
    ids=["named", "unnamed", "empty", "all-null", "nested"],
)
def test_series_round_trip(ctx_pair, series):
    result = round_trip(series, ctx_pair)
    assert isinstance(result, pl.Series)
    assert result.name == series.name
    assert result.dtype == series.dtype
    assert result.equals(series)


def test_series_payload_describes_the_series(ctx_pair, tmp_path):
    ser_ctx, _ = ctx_pair
    tree = to_json_tree(pl.Series("price", [1.0, 2.0]), ser_ctx)
    assert tree["__type__"] == "polars.Series"
    assert (tree["name"], tree["length"], tree["dtype"]) == ("price", 2, "Float64")
    assert [p.name.split("-")[0] for p in tmp_path.glob("*.parquet")] == ["series"]


class Holder:
    def __init__(self, series):
        self.series = series


def test_series_inside_containers_and_objects(ctx_pair):
    series = pl.Series("a", [1, 2])
    value = {"in_dict": series, "in_list": [series], "in_object": Holder(series)}
    result = round_trip(value, ctx_pair)
    assert result["in_dict"].equals(series)
    assert result["in_list"][0].equals(series)
    assert result["in_object"].series.equals(series)


def test_missing_series_file_is_a_corrupt_record(ctx_pair, tmp_path):
    ser_ctx, deser_ctx = ctx_pair
    tree = to_json_tree(pl.Series("a", [1]), ser_ctx)
    for path in tmp_path.glob("*.parquet"):
        path.unlink()
    with pytest.raises(CorruptRecordError, match="Series file is missing"):
        from_json_tree(tree, deser_ctx)
