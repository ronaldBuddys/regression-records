import datetime
from decimal import Decimal

import polars as pl
import pytest

from regression_records.comparison import compare
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


# ---------------------------------------------------------------------- dtype matrix

_MOMENT = datetime.datetime(2024, 3, 1, 12, 30, 15, 123456)

#: One column per dtype parquet is expected to hold, each with a null. A polars
#: upgrade that stops round-tripping one of them should fail here, by name.
DTYPE_COLUMNS = {
    **{
        str(dtype): pl.Series([1, None, 3], dtype=dtype)
        for dtype in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )
    },
    "Float32": pl.Series([1.5, None, float("nan")], dtype=pl.Float32),
    "Float64": pl.Series([1.5, None, float("inf")], dtype=pl.Float64),
    "Boolean": pl.Series([True, None, False]),
    "String": pl.Series(["a", None, ""]),
    "Binary": pl.Series([b"\x00\x01", None, b""]),
    "Date": pl.Series([datetime.date(2024, 3, 1), None, datetime.date(1970, 1, 1)]),
    **{
        f"Datetime[{unit}]": pl.Series(
            [_MOMENT, None, _MOMENT], dtype=pl.Datetime(unit)
        )
        for unit in ("ms", "us", "ns")
    },
    "Datetime[tz]": pl.Series(
        [_MOMENT, None, _MOMENT], dtype=pl.Datetime("us", "Europe/London")
    ),
    "Duration": pl.Series([datetime.timedelta(hours=3), None, datetime.timedelta(0)]),
    "Time": pl.Series([datetime.time(9, 30), None, datetime.time(0, 0)]),
    # Precision declared: before polars 1.35 an inferred one is None, and comes back
    # from parquet as 38 (see docs/limitations.md).
    "Decimal": pl.Series(
        [Decimal("1.10"), None, Decimal("-3.25")], dtype=pl.Decimal(10, 2)
    ),
    "Categorical": pl.Series(["x", None, "y"], dtype=pl.Categorical),
    "Enum": pl.Series(["lo", None, "hi"], dtype=pl.Enum(["lo", "hi"])),
    "List": pl.Series([[1.0, 2.0], None, []]),
    "Array": pl.Series([[1.0, 2.0], None, [3.0, 4.0]], dtype=pl.Array(pl.Float64, 2)),
    "Struct": pl.Series([{"x": 1, "y": "a"}, None, {"x": 2, "y": None}]),
    "List[Struct]": pl.Series([[{"x": 1.0}], None, [{"x": 2.0}, {"x": None}]]),
    "Null": pl.Series([None, None, None]),
}


@pytest.mark.parametrize("name", DTYPE_COLUMNS)
@pytest.mark.parametrize("as_series", [False, True], ids=["frame", "series"])
def test_every_dtype_round_trips_with_its_schema(ctx_pair, name, as_series):
    column = DTYPE_COLUMNS[name].alias(name)
    value = column if as_series else column.to_frame()
    result = round_trip(value, ctx_pair)
    assert (result.dtype if as_series else result.schema) == (
        column.dtype if as_series else value.schema
    )
    assert result.equals(value)
    assert compare(value, result)


def test_the_whole_dtype_matrix_round_trips_as_one_frame(ctx_pair):
    frame = pl.DataFrame([column.alias(name) for name, column in DTYPE_COLUMNS.items()])
    result = round_trip(frame, ctx_pair)
    assert result.schema == frame.schema
    assert result.equals(frame)


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame(schema={"a": pl.Int64, "b": pl.String, "c": pl.List(pl.Float64)}),
        pl.DataFrame(),
        pl.DataFrame({"a": [1], "b": ["x"]}),
    ],
    ids=["no-rows", "no-columns", "one-row"],
)
def test_degenerate_shapes_round_trip(ctx_pair, frame):
    result = round_trip(frame, ctx_pair)
    assert result.schema == frame.schema
    assert result.shape == frame.shape
    assert compare(frame, result)


def test_re_recording_a_frame_does_not_rewrite_its_file(ctx_pair, tmp_path):
    ser_ctx, _ = ctx_pair
    to_json_tree(pl.DataFrame({"a": [1, 2, 3]}), ser_ctx)
    (written,) = tmp_path.glob("*.parquet")
    before = written.stat().st_mtime_ns
    to_json_tree(pl.DataFrame({"a": [1, 2, 3]}), ser_ctx)
    assert written.stat().st_mtime_ns == before


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
