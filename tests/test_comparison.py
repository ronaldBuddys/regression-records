import numpy as np
import pandas as pd
import polars as pl
import pytest

from regression_records.comparison import ROOT_PATH, ToleranceConfig, compare
from regression_records.comparison.base import ComparatorRegistry, Diff
from regression_records.comparison.numpy_comparator import NumpyArrayComparator
from regression_records.comparison.objects import GenericObjectComparator


def kinds(result):
    return [d.kind for d in result.diffs]


def paths(result):
    return [d.path for d in result.diffs]


# --------------------------------------------------------------------------- scalars


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 1, -3, 1.5, "hello", "", b"\x00\x01", 3 + 4j],
)
def test_identical_scalars_pass(value):
    assert compare(value, value)


@pytest.mark.parametrize(
    "expected,actual,kind",
    [
        (1, 2, "value_mismatch"),
        (1.0, 1.5, "value_mismatch"),
        ("a", "b", "value_mismatch"),
        (True, False, "value_mismatch"),
        (1, "1", "type_mismatch"),
        (None, 0, "type_mismatch"),
        (True, 1, "type_mismatch"),
    ],
)
def test_mismatched_scalars(expected, actual, kind):
    result = compare(expected, actual)
    assert not result
    assert kinds(result) == [kind]
    assert paths(result) == [ROOT_PATH]


def test_float_within_default_tolerance_passes():
    assert compare(1.0, 1.0 + 1e-12)


def test_float_outside_default_tolerance_fails():
    result = compare(1.0, 1.001)
    assert not result
    assert result.diffs[0].tolerance_used.max_abs_diff == pytest.approx(0.001)


def test_atol_allows_small_absolute_difference():
    tol = ToleranceConfig(rtol=0.0, atol=1e-3)
    assert compare(1.0, 1.0005, tol)
    assert not compare(1.0, 1.005, tol)


def test_nan_equality_is_configurable():
    assert compare(float("nan"), float("nan"))
    assert not compare(float("nan"), float("nan"), ToleranceConfig(equal_nan=False))


def test_infinities_compare_exactly():
    assert compare(float("inf"), float("inf"))
    assert not compare(float("inf"), float("-inf"))
    assert not compare(float("inf"), 1e308)


def test_numpy_scalar_dtype_change_is_reported():
    result = compare(np.float64(1.0), np.float32(1.0))
    assert not result
    assert kinds(result) == ["dtype_mismatch"]
    assert compare(
        np.float64(1.0), np.float32(1.0), ToleranceConfig(strict_dtype=False)
    )


def test_numpy_scalar_values_use_tolerance():
    assert compare(np.float64(1.0), np.float64(1.0 + 1e-12))
    result = compare(np.float64(1.0), np.float64(1.5))
    assert kinds(result) == ["value_mismatch"]
    assert result.diffs[0].tolerance_used is not None


def test_numpy_scalar_against_a_plain_python_number_compares_numerically():
    # Only one side is a numpy scalar, so strict_dtype has no two dtypes to compare and
    # the values are simply checked against each other.
    assert compare(np.float64(1.0), 1.0)
    assert kinds(compare(np.float64(1.0), 2.0)) == ["value_mismatch"]


@pytest.mark.parametrize(
    "value", [np.bool_(True), np.str_("hello"), np.datetime64("2024-01-01")]
)
def test_non_numeric_numpy_scalars_compare_exactly(value):
    assert compare(value, value)


def test_numpy_bool_change_is_a_value_mismatch_not_a_numeric_one():
    result = compare(np.bool_(True), np.bool_(False))
    assert kinds(result) == ["value_mismatch"]
    # True is not "1, within tolerance" - no tolerance was involved in deciding this.
    assert result.diffs[0].tolerance_used is None


# ------------------------------------------------------------------------ structures


def test_dict_missing_and_extra_keys_with_paths():
    result = compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
    assert not result
    assert sorted(zip(kinds(result), paths(result))) == [
        ("extra_key", "$.c"),
        ("missing_key", "$.b"),
    ]


def test_dict_collects_all_diffs_without_short_circuiting():
    result = compare({"a": 1, "b": 2, "c": 3}, {"a": 9, "b": 8, "c": 7})
    assert len(result.diffs) == 3
    assert paths(result) == ["$.a", "$.b", "$.c"]


def test_nested_dict_paths():
    result = compare({"outer": {"inner": [1, 2, 3]}}, {"outer": {"inner": [1, 99, 3]}})
    assert paths(result) == ["$.outer.inner[1]"]


def test_list_length_mismatch_short_circuits_elements():
    result = compare([1, 2, 3], [1, 2])
    assert kinds(result) == ["length_mismatch"]


def test_list_and_tuple_are_not_interchangeable():
    result = compare((1, 2), [1, 2])
    assert kinds(result) == ["type_mismatch"]


def test_tuple_elements_compared_by_position():
    result = compare((1, 2), (1, 5))
    assert paths(result) == ["$[1]"]


def test_set_membership_diffs():
    result = compare({1, 2, 3}, {1, 2, 4})
    assert sorted(kinds(result)) == ["extra_key", "missing_key"]


def test_dict_vs_list_is_a_type_mismatch():
    assert kinds(compare({"a": 1}, [1])) == ["type_mismatch"]


# ----------------------------------------------------------------------------- numpy


def test_identical_arrays_pass():
    assert compare(np.arange(10), np.arange(10))


def test_array_shape_mismatch_short_circuits():
    result = compare(np.zeros((2, 3)), np.zeros((3, 2)))
    assert kinds(result) == ["shape_mismatch"]


def test_array_dtype_mismatch_reported_and_values_still_compared():
    result = compare(np.array([1, 2], dtype=np.int32), np.array([1, 5], dtype=np.int64))
    assert kinds(result) == ["dtype_mismatch", "value_mismatch"]


def test_array_dtype_mismatch_suppressed_when_not_strict():
    same_values = (np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int64))
    assert not compare(*same_values)
    assert compare(*same_values, ToleranceConfig(strict_dtype=False))


def test_array_tolerance_pass_and_fail():
    expected = np.array([1.0, 2.0, 3.0])
    assert compare(expected, expected + 1e-12)
    result = compare(expected, expected + 0.1)
    assert not result
    assert kinds(result) == ["value_mismatch"]
    assert result.diffs[0].tolerance_used.max_abs_diff == pytest.approx(0.1)


def test_array_diff_message_reports_count_and_indices():
    expected = np.zeros(5)
    actual = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    result = compare(expected, actual)
    assert "2 of 5 element(s) differ" in result.diffs[0].message
    assert "1, 3" in result.diffs[0].message


def test_array_index_reporting_is_capped():
    result = compare(np.zeros(1000), np.ones(1000))
    assert "1000 of 1000" in result.diffs[0].message
    assert "980 more" in result.diffs[0].message


def test_array_nan_handling():
    expected = np.array([1.0, np.nan])
    assert compare(expected, np.array([1.0, np.nan]))
    assert not compare(
        expected, np.array([1.0, np.nan]), ToleranceConfig(equal_nan=False)
    )


def test_string_array_comparison():
    assert compare(np.array(["a", "b"]), np.array(["a", "b"]))
    assert not compare(np.array(["a", "b"]), np.array(["a", "c"]))


def test_array_versus_non_array_is_a_type_mismatch():
    assert kinds(compare(np.arange(3), [0, 1, 2])) == ["type_mismatch"]


def test_multidimensional_index_reporting():
    expected = np.zeros((2, 2))
    actual = np.array([[0.0, 0.0], [0.0, 1.0]])
    result = compare(expected, actual)
    assert "(1, 1)" in result.diffs[0].message


# ---------------------------------------------------------------------------- polars


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


# ---------------------------------------------------------------------------- pandas


def test_identical_pandas_frames_pass():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert compare(df, df.copy())


def test_pandas_schema_mismatch_short_circuits():
    result = compare(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert kinds(result) == ["schema_mismatch"]


def test_pandas_index_mismatch_short_circuits_values():
    expected = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
    actual = pd.DataFrame({"a": [1, 2]}, index=[0, 5])
    result = compare(expected, actual)
    assert kinds(result) == ["value_mismatch"]
    assert paths(result) == ["$.index"]


def test_pandas_numeric_column_uses_tolerance():
    expected = pd.DataFrame({"a": [1.0, 2.0]})
    assert compare(expected, pd.DataFrame({"a": [1.0 + 1e-12, 2.0]}))
    result = compare(expected, pd.DataFrame({"a": [1.0, 9.0]}))
    assert paths(result) == ["$['a']"]


def test_pandas_object_column_nan_treated_as_equal():
    expected = pd.DataFrame({"a": ["x", None]})
    assert compare(expected, pd.DataFrame({"a": ["x", None]}))


def test_pandas_series_round_trip_and_mismatch():
    expected = pd.Series([1.0, 2.0], name="s")
    assert compare(expected, expected.copy())
    result = compare(expected, pd.Series([1.0, 3.0], name="s"))
    assert kinds(result) == ["value_mismatch"]


def test_pandas_series_name_change_is_a_schema_mismatch():
    result = compare(pd.Series([1], name="a"), pd.Series([1], name="b"))
    assert kinds(result) == ["schema_mismatch"]


# --------------------------------------------------------------------------- objects


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Widget:
    def __init__(self, label, values):
        self.label = label
        self.values = values

    def __getstate__(self):
        return {"label": self.label, "values": self.values}


class Opaque:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def test_object_attributes_compared_by_path():
    result = compare(Point(1, 2), Point(1, 5))
    assert kinds(result) == ["value_mismatch"]
    assert paths(result) == ["$.y"]


def test_object_of_different_class_is_a_type_mismatch():
    assert kinds(compare(Point(1, 2), Widget("a", []))) == ["type_mismatch"]


def test_getstate_based_object_comparison():
    assert compare(Widget("a", [1, 2]), Widget("a", [1, 2]))
    result = compare(Widget("a", [1, 2]), Widget("b", [1, 2]))
    assert paths(result) == ["$.label"]


def test_nested_object_holding_an_array():
    expected = Point(np.array([1.0, 2.0]), {"inner": Point(1, 2)})
    actual = Point(np.array([1.0, 2.5]), {"inner": Point(1, 3)})
    result = compare(expected, actual)
    assert sorted(paths(result)) == ["$.x", "$.y.inner.y"]


def test_object_without_inspectable_state_falls_back_to_equality():
    assert compare(Opaque(1), Opaque(1)) is not None
    result = compare(Opaque(1), Opaque(2))
    assert kinds(result) == ["value_mismatch"]
    assert "no inspectable state" in result.diffs[0].message


def test_datetime_is_compared_by_value_not_state():
    import datetime

    assert compare(datetime.date(2026, 1, 1), datetime.date(2026, 1, 1))
    result = compare(datetime.date(2026, 1, 1), datetime.date(2026, 1, 2))
    assert kinds(result) == ["value_mismatch"]


# ------------------------------------------------------------------ tolerance config


def test_path_overrides_apply_to_a_subtree():
    tol = ToleranceConfig(
        rtol=0.0, path_overrides={"$.loose": ToleranceConfig(atol=1.0)}
    )
    expected = {"loose": {"a": 1.0}, "strict": 1.0}
    assert compare(expected, {"loose": {"a": 1.5}, "strict": 1.0}, tol)
    result = compare(expected, {"loose": {"a": 1.5}, "strict": 1.5}, tol)
    assert paths(result) == ["$.strict"]


def test_most_specific_path_override_wins():
    tol = ToleranceConfig(
        rtol=0.0,
        path_overrides={
            "$.a": ToleranceConfig(atol=10.0),
            "$.a.b": ToleranceConfig(atol=0.0),
        },
    )
    result = compare({"a": {"b": 1.0, "c": 1.0}}, {"a": {"b": 2.0, "c": 2.0}}, tol)
    assert paths(result) == ["$.a.b"]


def test_path_override_applies_to_indexed_children():
    tol = ToleranceConfig(
        rtol=0.0, path_overrides={"$.vals": ToleranceConfig(atol=1.0)}
    )
    assert compare({"vals": [1.0, 2.0]}, {"vals": [1.5, 2.5]}, tol)


@pytest.mark.parametrize(
    "frame",
    [
        lambda data: pl.DataFrame(data),
        lambda data: pd.DataFrame(data),
    ],
    ids=["polars", "pandas"],
)
def test_a_path_override_can_name_a_single_dataframe_column(frame):
    """A dataframe comparator walks its own columns instead of recursing through the
    dispatcher, so per-column overrides only work if it resolves them itself."""
    tol = ToleranceConfig(
        rtol=0.0,
        path_overrides={"$['loose']": ToleranceConfig(atol=1.0)},
    )
    expected = frame({"loose": [1.0, 2.0], "strict": [1.0, 2.0]})

    assert compare(expected, frame({"loose": [1.5, 2.5], "strict": [1.0, 2.0]}), tol)

    result = compare(expected, frame({"loose": [1.5, 2.5], "strict": [1.5, 2.0]}), tol)
    assert paths(result) == ["$['strict']"]


# ------------------------------------------------------------------------- machinery


def test_result_summary_lists_diffs_and_caps_them():
    result = compare({str(i): i for i in range(30)}, {str(i): i + 1 for i in range(30)})
    summary = result.summary(max_diffs=5)
    assert "comparison failed with 30 diff(s)" in summary
    assert "... and 25 more" in summary
    assert compare(1, 1).summary() == "comparison passed"


def test_diffs_hold_reprs_not_live_objects():
    result = compare(np.zeros(3), np.ones(3))
    assert isinstance(result.diffs[0].expected, str)
    assert isinstance(result.diffs[0].actual, str)


def test_custom_comparator_registry_is_honoured():
    class AlwaysDiffers:
        def can_handle(self, expected, actual):
            return isinstance(expected, int)

        def compare(self, expected, actual, path, tolerance, compare_fn, diffs):
            diffs.append(
                Diff(
                    path=path,
                    kind="value_mismatch",
                    expected="e",
                    actual="a",
                    message="nope",
                )
            )

    registry = ComparatorRegistry()
    registry.register(AlwaysDiffers())
    assert not compare(1, 1, registry=registry)
    assert compare("same", "same", registry=registry)


def test_a_registered_comparator_beats_the_fallback_but_not_the_specific_ones():
    class ClaimsEverything:
        def can_handle(self, expected, actual):
            return True

        def compare(
            self, expected, actual, path, tolerance, compare_fn, diffs
        ):  # pragma: no cover
            pass

    specific, custom, fallback = (
        NumpyArrayComparator(),
        ClaimsEverything(),
        GenericObjectComparator(),
    )
    registry = ComparatorRegistry()
    registry.register(specific)
    registry.register(fallback, fallback=True)
    registry.register(custom)

    assert registry.find(np.zeros(2), np.zeros(2)) is specific
    assert registry.find(Point(1, 2), Point(1, 2)) is custom


def test_registering_on_a_copy_leaves_the_original_alone():
    class ClaimsEverything:
        def can_handle(self, expected, actual):
            return True

        def compare(
            self, expected, actual, path, tolerance, compare_fn, diffs
        ):  # pragma: no cover
            pass

    registry = ComparatorRegistry()
    registry.register(GenericObjectComparator(), fallback=True)

    clone = registry.copy()
    custom = ClaimsEverything()
    clone.register(custom)

    assert clone.find(1, 1) is custom
    assert registry.find(1, 1) is not custom
    assert len(registry) == 1


def test_comparison_never_raises_on_mismatch():
    # compare() is a pure function; raising is a decorator-layer policy.
    result = compare({"a": np.zeros(2)}, {"a": "not an array"})
    assert not result
    assert kinds(result) == ["type_mismatch"]
