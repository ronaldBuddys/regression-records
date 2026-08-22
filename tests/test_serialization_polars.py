import datetime

import polars as pl
import pytest

from regression_records.exceptions import SerializationError
from regression_records.serialization import to_json_tree

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


def test_nested_dataframe_in_dict(ctx_pair):
    value = {"df": pl.DataFrame({"a": [1, 2]}), "meta": {"rows": 2}}
    result = round_trip(value, ctx_pair)
    assert result["df"].equals(value["df"])
    assert result["meta"] == {"rows": 2}
