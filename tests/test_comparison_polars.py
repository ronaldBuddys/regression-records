import datetime

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
    message = result.diffs[-1].message
    assert "reporting stopped after 20 differing columns" in message
    # Column 20 differed; the 9 after it were never compared, so may not differ.
    assert (
        "column 'c20' also differs, and the 9 column(s) after it were not compared"
        in message
    )


def test_polars_versus_non_frame_is_a_type_mismatch():
    assert kinds(compare(pl.DataFrame({"a": [1]}), {"a": [1]})) == ["type_mismatch"]


def test_polars_row_diff_reporting_is_capped():
    expected = pl.DataFrame({"a": [0.0] * 50, "s": ["x"] * 50})
    actual = pl.DataFrame({"a": [1.0] * 50, "s": ["y"] * 50})
    result = compare(expected, actual)
    assert paths(result) == ["$['a']", "$['s']"]
    for diff in result.diffs:
        assert "50 of 50" in diff.message
        assert "... (30 more)" in diff.message


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame(schema={"a": pl.Float64, "s": pl.String}),
        pl.DataFrame(),
        pl.DataFrame({"a": [1.0]}),
    ],
    ids=["no-rows", "no-columns", "one-row"],
)
def test_polars_degenerate_frames_compare_equal_to_themselves(frame):
    assert compare(frame, frame.clone())


# ---------------------------------------------------------------------------- schema

_NOON = [datetime.datetime(2024, 1, 1, 12)]

DTYPE_CHANGES = {
    "int-widened": (
        pl.Series([1, 2], dtype=pl.Int32),
        pl.Series([1, 2], dtype=pl.Int64),
    ),
    "float-widened": (
        pl.Series([1.5, 2.5], dtype=pl.Float32),
        pl.Series([1.5, 2.5], dtype=pl.Float64),
    ),
    "categorical-to-string": (
        pl.Series(["a", "b"], dtype=pl.Categorical),
        pl.Series(["a", "b"]),
    ),
    "enum-categories": (
        pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])),
        pl.Series(["a", "b"], dtype=pl.Enum(["a", "b", "c"])),
    ),
    "time-zone": (
        pl.Series(_NOON, dtype=pl.Datetime("us", "UTC")),
        pl.Series(_NOON, dtype=pl.Datetime("us", "Europe/Paris")),
    ),
    "time-unit": (
        pl.Series(_NOON, dtype=pl.Datetime("us")),
        pl.Series(_NOON, dtype=pl.Datetime("ns")),
    ),
}


@pytest.mark.parametrize("change", DTYPE_CHANGES)
def test_polars_dtype_changes_are_schema_mismatches(change):
    before, after = DTYPE_CHANGES[change]
    result = compare(pl.DataFrame({"c": before}), pl.DataFrame({"c": after}))
    assert kinds(result) == ["schema_mismatch"]
    assert "dtype change(s)" in result.diffs[0].message
    assert kinds(compare(before.alias("c"), after.alias("c"))) == ["schema_mismatch"]


@pytest.mark.parametrize("change", ["int-widened", "float-widened"])
def test_polars_widened_numbers_pass_without_strict_dtype(change):
    before, after = DTYPE_CHANGES[change]
    relaxed = ToleranceConfig(strict_dtype=False)
    assert compare(pl.DataFrame({"c": before}), pl.DataFrame({"c": after}), relaxed)
    assert not compare(
        pl.DataFrame({"c": before}), pl.DataFrame({"c": after + 1}), relaxed
    )


@pytest.mark.parametrize("change", DTYPE_CHANGES)
def test_polars_any_dtype_change_compares_without_raising_when_relaxed(change):
    """compare() is a diagnostic: whatever polars makes of the pair, it reports."""
    before, after = DTYPE_CHANGES[change]
    result = compare(
        pl.DataFrame({"c": before}),
        pl.DataFrame({"c": after}),
        ToleranceConfig(strict_dtype=False),
    )
    assert set(kinds(result)) <= {"value_mismatch"}


def test_polars_renamed_column_names_both_columns():
    result = compare(pl.DataFrame({"px": [1.0]}), pl.DataFrame({"price": [1.0]}))
    assert "missing column(s) ['px']" in result.diffs[0].message
    assert "unexpected column(s) ['price']" in result.diffs[0].message


# ------------------------------------------------------------------------- tolerance


def test_polars_a_column_override_narrows_a_broader_one():
    expected = pl.DataFrame({"id": [1, 2], "price": [10.0, 20.0], "qty": [1.0, 2.0]})
    actual = pl.DataFrame({"id": [1, 3], "price": [10.4, 20.4], "qty": [1.4, 2.4]})
    tol = ToleranceConfig(
        path_overrides={
            "$.output": ToleranceConfig(atol=0.5),
            "$.output['qty']": ToleranceConfig(atol=0.1),
            "$.output['id']": ToleranceConfig(rtol=0, atol=0),
        }
    )
    result = compare(expected, actual, tol, root="$.output")
    assert paths(result) == ["$.output['id']", "$.output['qty']"]


# ----------------------------------------------------------------------------- nulls

NAN = float("nan")


def test_polars_null_and_nan_are_different():
    result = compare(pl.DataFrame({"a": [1.0, None]}), pl.DataFrame({"a": [1.0, NAN]}))
    assert kinds(result) == ["value_mismatch"]
    assert "1 of 2 value(s) are null on one side only at row(s): 1" in (
        result.diffs[0].message
    )


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ([1.0, None], [1.0, 2.0]),
        ([1, None], [1, 2]),
        (["a", None], ["a", "b"]),
        ([[1.0], None], [[1.0], [2.0]]),
        ([{"x": 1}, None], [{"x": 1}, {"x": 2}]),
    ],
    ids=["float", "int", "string", "list", "struct"],
)
def test_polars_null_against_a_value_is_reported_once(expected, actual):
    result = compare(pl.DataFrame({"a": expected}), pl.DataFrame({"a": actual}))
    assert kinds(result) == ["value_mismatch"]
    assert "null on one side only at row(s): 1" in result.diffs[0].message


@pytest.mark.parametrize(
    "values",
    [[None, None], [1.0, None], [1, None], ["a", None], [[1.0, None], None]],
    ids=["all-null", "float", "int", "string", "list"],
)
def test_polars_nulls_in_the_same_places_are_equal(values):
    assert compare(pl.DataFrame({"a": values}), pl.DataFrame({"a": values}))


def test_polars_nulls_do_not_stop_the_other_rows_using_tolerance():
    expected = pl.DataFrame({"a": [1.0, None, 3.0]})
    assert compare(expected, pl.DataFrame({"a": [1.0 + 1e-12, None, 3.0]}))

    result = compare(expected, pl.DataFrame({"a": [None, None, 3.5]}))
    assert kinds(result) == ["value_mismatch", "value_mismatch"]
    assert "row(s): 0" in result.diffs[0].message
    assert "index/indices: 2" in result.diffs[1].message


def test_polars_nan_against_nan_follows_equal_nan():
    frame = pl.DataFrame({"a": [1.0, NAN]})
    assert compare(frame, frame.clone())
    result = compare(frame, frame.clone(), ToleranceConfig(equal_nan=False))
    assert kinds(result) == ["value_mismatch"]


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt64])
def test_polars_big_integers_beside_nulls_are_compared_exactly(dtype):
    """A null used to send the column through float64, where 2**60 + 1 == 2**60."""
    expected = pl.DataFrame({"id": pl.Series([2**60 + 1, None, 7], dtype=dtype)})
    actual = pl.DataFrame({"id": pl.Series([2**60, None, 7], dtype=dtype)})
    result = compare(expected, actual)
    assert kinds(result) == ["value_mismatch"]
    assert "index/indices: 0" in result.diffs[0].message
    assert compare(expected, expected.clone())


def test_polars_small_integers_keep_their_tolerance():
    # The documented (if debatable) policy: integers are compared like floats.
    expected = pl.DataFrame({"n": [1_000_000_000, None]})
    assert compare(expected, pl.DataFrame({"n": [1_000_000_050, None]}))
    exact = ToleranceConfig(path_overrides={"$['n']": ToleranceConfig(rtol=0, atol=0)})
    assert not compare(expected, pl.DataFrame({"n": [1_000_000_050, None]}), exact)


def test_polars_decimal_columns_compare_exactly():
    from decimal import Decimal

    expected = pl.DataFrame({"d": [Decimal("1.10"), None]})
    assert compare(expected, expected.clone())
    result = compare(expected, pl.DataFrame({"d": [Decimal("1.11"), None]}))
    assert kinds(result) == ["value_mismatch"]
    assert "differ at row(s): 0" in result.diffs[0].message


# ---------------------------------------------------------------------------- nested

ARRAY = pl.Array(pl.Float64, 2)


@pytest.mark.parametrize(
    ("make", "dtype"),
    [
        (lambda x: [[1.0, x], [3.0], [], None], None),
        (lambda x: [[1.0, x], [3.0, 4.0]], ARRAY),
        (lambda x: [{"v": x, "tag": "a"}, {"v": 2.0, "tag": "b"}], None),
        (lambda x: [[{"v": x}], [{"v": 2.0}, {"v": None}]], None),
        (lambda x: [[[1.0, x]], [[3.0], []]], None),
    ],
    ids=["list", "array", "struct", "list-of-struct", "list-of-list"],
)
def test_polars_tolerance_reaches_inside_nested_columns(make, dtype):
    def frame(x):
        return pl.DataFrame({"n": pl.Series(make(x), dtype=dtype)})

    expected = frame(2.0)
    assert compare(expected, frame(2.0 + 1e-12))

    result = compare(expected, frame(2.5))
    assert kinds(result) == ["value_mismatch"]
    assert "1 of" in result.diffs[0].message
    assert "nested value(s) differ beyond tolerance at row(s): 0" in (
        result.diffs[0].message
    )
    assert result.diffs[0].tolerance_used is not None

    loose = ToleranceConfig(path_overrides={"$['n']": ToleranceConfig(atol=1.0)})
    assert compare(expected, frame(2.5), loose)


def test_polars_nested_lists_of_different_length_differ():
    result = compare(
        pl.DataFrame({"n": [[1.0], [2.0, 3.0], []]}),
        pl.DataFrame({"n": [[1.0], [2.0], [0.0]]}),
    )
    assert kinds(result) == ["value_mismatch"]
    assert "2 of 3 nested value(s)" in result.diffs[0].message
    assert "row(s): 1, 2" in result.diffs[0].message


def test_polars_nested_nulls_and_nans_are_told_apart():
    expected = pl.DataFrame({"n": [[1.0, None], [NAN]]})
    assert compare(expected, expected.clone())
    assert not compare(expected, expected.clone(), ToleranceConfig(equal_nan=False))

    result = compare(expected, pl.DataFrame({"n": [[1.0, NAN], [NAN]]}))
    assert "row(s): 0" in result.diffs[0].message


def test_polars_exact_leaves_beside_float_ones_stay_exact():
    expected = pl.DataFrame({"s": [{"v": 1.0, "tag": "a"}]})
    result = compare(expected, pl.DataFrame({"s": [{"v": 1.0, "tag": "b"}]}))
    assert kinds(result) == ["value_mismatch"]
    loose = ToleranceConfig(rtol=1.0, atol=1.0)
    assert not compare(expected, pl.DataFrame({"s": [{"v": 1.0, "tag": "b"}]}), loose)


def test_polars_nested_series_use_tolerance_too():
    expected = pl.Series("n", [[1.0, 2.0]])
    assert compare(expected, pl.Series("n", [[1.0, 2.0 + 1e-12]]))
    assert not compare(expected, pl.Series("n", [[1.0, 2.5]]))


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
