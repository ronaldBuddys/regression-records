import pandas as pd
import pytest

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


def test_mixed_dtype_dataframe_falls_back_to_pickle(ctx_pair):
    df = pd.DataFrame({"mixed": [1, "two", 3.0, None]})
    with pytest.warns(UserWarning, match="falling back to pickle"):
        result = round_trip(df, ctx_pair)
    pd.testing.assert_frame_equal(result, df)


def test_nested_dataframe_in_dict(ctx_pair):
    value = {"df": pd.DataFrame({"a": [1, 2]}), "meta": {"rows": 2}}
    result = round_trip(value, ctx_pair)
    pd.testing.assert_frame_equal(result["df"], value["df"])
    assert result["meta"] == {"rows": 2}
