import polars as pl
import pytest

from regression_records.comparison import ToleranceConfig, compare


def kinds(result):
    return [d.kind for d in result.diffs]


def paths(result):
    return [d.path for d in result.diffs]


# ------------------------------------------------------------------------ dataframes


def test_identical_polars_frames_pass():
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert compare(df, df.clone())


def test_polars_schema_mismatch_short_circuits():
    result = compare(pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [1]}))
    assert kinds(result) == ["schema_mismatch"]
    assert "missing column(s) ['a']" in result.diffs[0].message
    assert "unexpected column(s) ['b']" in result.diffs[0].message


def test_polars_dtype_change_is_a_schema_mismatch():
    result = compare(
        pl.DataFrame({"a": [1]}, schema={"a": pl.Int32}),
        pl.DataFrame({"a": [1]}, schema={"a": pl.Int64}),
    )
    assert kinds(result) == ["schema_mismatch"]
    assert "dtype change(s)" in result.diffs[0].message


@pytest.mark.parametrize("strict_dtype", [True, False])
def test_polars_reordered_columns_are_a_schema_mismatch(strict_dtype):
    result = compare(
        pl.DataFrame({"x": [1], "y": [2]}),
        pl.DataFrame({"y": [2], "x": [1]}),
        ToleranceConfig(strict_dtype=strict_dtype),
    )
    assert kinds(result) == ["schema_mismatch"]
    assert "column order differs" in result.diffs[0].message


def test_polars_row_count_mismatch():
    result = compare(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [1]}))
    assert kinds(result) == ["length_mismatch"]


def test_polars_numeric_column_uses_tolerance():
    expected = pl.DataFrame({"a": [1.0, 2.0]})
    assert compare(expected, pl.DataFrame({"a": [1.0 + 1e-12, 2.0]}))
    result = compare(expected, pl.DataFrame({"a": [1.5, 2.0]}))
    assert kinds(result) == ["value_mismatch"]
    assert paths(result) == ["$['a']"]


def test_polars_string_column_mismatch_reports_rows():
    result = compare(
        pl.DataFrame({"s": ["a", "b", "c"]}), pl.DataFrame({"s": ["a", "z", "c"]})
    )
    assert kinds(result) == ["value_mismatch"]
    assert "1 of 3 value(s) differ at row(s): 1" in result.diffs[0].message


def test_polars_null_difference_detected():
    result = compare(pl.DataFrame({"s": ["a", None]}), pl.DataFrame({"s": ["a", "b"]}))
    assert kinds(result) == ["value_mismatch"]


def test_polars_nested_column_mismatch():
    result = compare(
        pl.DataFrame({"n": [[1, 2], [3]]}), pl.DataFrame({"n": [[1, 2], [4]]})
    )
    assert kinds(result) == ["value_mismatch"]


def test_polars_column_diff_reporting_is_capped():
    wide_expected = pl.DataFrame({f"c{i}": [0.0] for i in range(30)})
    wide_actual = pl.DataFrame({f"c{i}": [1.0] for i in range(30)})
    result = compare(wide_expected, wide_actual)
    assert len(result.diffs) == 21
    assert "reporting stopped after 20 differing columns" in result.diffs[-1].message


def test_polars_versus_non_frame_is_a_type_mismatch():
    assert kinds(compare(pl.DataFrame({"a": [1]}), {"a": [1]})) == ["type_mismatch"]


# ---------------------------------------------------------------------------- series


def test_identical_polars_series_pass():
    series = pl.Series("a", [1.0, None, 3.0])
    assert compare(series, series.clone())


def test_polars_series_name_change_is_a_schema_mismatch():
    result = compare(pl.Series("a", [1]), pl.Series("b", [1]))
    assert kinds(result) == ["schema_mismatch"]
    assert "name='a'" in result.diffs[0].expected


def test_polars_series_dtype_change_respects_strict_dtype():
    expected = pl.Series("a", [1, 2], dtype=pl.Int32)
    actual = pl.Series("a", [1, 2], dtype=pl.Int64)
    assert kinds(compare(expected, actual)) == ["schema_mismatch"]
    assert compare(expected, actual, ToleranceConfig(strict_dtype=False))


def test_polars_series_length_mismatch():
    result = compare(pl.Series("a", [1, 2]), pl.Series("a", [1]))
    assert kinds(result) == ["length_mismatch"]


def test_polars_series_values_use_tolerance_and_report_rows():
    expected = pl.Series("a", [1.0, 2.0, 3.0])
    assert compare(expected, pl.Series("a", [1.0, 2.0 + 1e-12, 3.0]))
    result = compare(expected, pl.Series("a", [1.0, 2.5, 3.0]), root="$.output")
    assert kinds(result) == ["value_mismatch"]
    assert paths(result) == ["$.output"]
    assert "index/indices: 1" in result.diffs[0].message
    assert result.diffs[0].tolerance_used is not None


def test_polars_series_string_values_report_rows():
    result = compare(pl.Series("s", ["a", "b"]), pl.Series("s", ["a", "z"]))
    assert "1 of 2 value(s) differ at row(s): 1" in result.diffs[0].message


def test_polars_series_versus_non_series_is_a_type_mismatch():
    assert kinds(compare(pl.Series("a", [1]), [1])) == ["type_mismatch"]
    assert kinds(compare([1], pl.Series("a", [1]))) == ["type_mismatch"]
