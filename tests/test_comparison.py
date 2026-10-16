import copy
import dataclasses

import numpy as np
import pandas as pd
import polars as pl
import pytest

from regression_records.comparison import ROOT_PATH, ToleranceConfig, compare
from regression_records.comparison.base import ComparatorRegistry, Diff
from regression_records.comparison.numpy_comparator import NumpyArrayComparator


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


# NaT is the datetime NaN: it has no value, so (like NaN) it matches only another NaT at
# the same position, and only under ``equal_nan``. Oracle: comparing a value with an
# independent copy of itself passes.
NAT_CASES = [
    pytest.param(np.array(["2024-01-01", "NaT"], "datetime64[D]"), id="datetime64-D"),
    pytest.param(
        np.array(["2024-01-01T00:00", "NaT"], "datetime64[ns]"), id="datetime64-ns"
    ),
    pytest.param(np.array([1, "NaT"], "timedelta64[D]"), id="timedelta64-D"),
    pytest.param(np.datetime64("NaT", "D"), id="datetime64-scalar"),
    pytest.param(np.timedelta64("NaT", "D"), id="timedelta64-scalar"),
    pytest.param([np.array(["NaT"], "datetime64[D]")], id="in-list"),
]


def _copy(value):
    return copy.deepcopy(value)


@pytest.mark.parametrize("value", NAT_CASES)
def test_nat_matches_nat(value):
    assert compare(value, _copy(value))


@pytest.mark.parametrize("value", NAT_CASES)
def test_nat_respects_equal_nan_false(value):
    assert not compare(value, _copy(value), ToleranceConfig(equal_nan=False))


def test_nat_against_a_date_differs():
    with_nat = np.array(["2024-01-01", "NaT"], "datetime64[D]")
    with_date = np.array(["2024-01-01", "2024-01-02"], "datetime64[D]")
    for expected, actual in [(with_nat, with_date), (with_date, with_nat)]:
        result = compare(expected, actual)
        assert kinds(result) == ["value_mismatch"]
        assert "index/indices: 1" in result.diffs[0].message
    date = np.datetime64("2024-01-02")
    assert not compare(np.datetime64("NaT", "D"), date)
    assert not compare(date, np.datetime64("NaT", "D"))


def test_nat_at_different_positions_differs():
    nat_first = np.array(["NaT", "2024-01-01"], "datetime64[D]")
    nat_last = np.array(["2024-01-01", "NaT"], "datetime64[D]")
    assert not compare(nat_first, nat_last)


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
    assert sorted(zip(kinds(result), paths(result), strict=True)) == [
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


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_arrays_beyond_float_precision_are_compared_exactly(dtype):
    """np.isclose works in floats, where 2**60 + 1 == 2**60 even at zero tolerance."""
    expected = np.array([2**60 + 1, 5], dtype=dtype)
    actual = np.array([2**60, 5], dtype=dtype)
    for tol in (ToleranceConfig(), ToleranceConfig(rtol=0, atol=0)):
        result = compare(expected, actual, tol)
        assert kinds(result) == ["value_mismatch"]
        assert "index/indices: 0" in result.diffs[0].message
    assert compare(expected, expected.copy())


def test_the_most_negative_int64_does_not_overflow_the_exactness_check():
    lowest = np.iinfo(np.int64).min
    assert compare(np.array([lowest]), np.array([lowest]))
    assert not compare(np.array([lowest]), np.array([lowest + 1]))


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


class Slotted:
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


def test_a_slots_object_is_compared_by_its_slots():
    assert compare(Slotted(1), Slotted(1))
    result = compare(Slotted(1), Slotted(2))
    assert paths(result) == ["$.value"]


def test_a_value_with_no_attributes_to_walk_is_compared_by_equality():
    import threading

    lock = threading.Lock()
    assert compare(lock, lock)
    result = compare(lock, threading.Lock())
    assert kinds(result) == ["value_mismatch"]


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


# Every inheritable field with a non-default parent value and a different override.
INHERITABLE = {
    "rtol": (1e-3, 0.5),
    "atol": (1e-3, 0.5),
    "equal_nan": (False, True),
    "strict_dtype": (False, True),
}


@pytest.mark.parametrize("inherited", list(INHERITABLE))
def test_an_override_inherits_the_fields_it_does_not_set(inherited):
    parent_values = {name: parent for name, (parent, _) in INHERITABLE.items()}
    # The override sets every field except the one under test.
    override = ToleranceConfig(
        **{name: value for name, (_, value) in INHERITABLE.items() if name != inherited}
    )
    tol = ToleranceConfig(**parent_values, path_overrides={"$.x": override})

    effective = tol.for_path("$.x.y")

    assert getattr(effective, inherited) == parent_values[inherited]
    for name, (_, value) in INHERITABLE.items():
        if name != inherited:
            assert getattr(effective, name) == value


@pytest.mark.parametrize(
    ("name", "parent", "default"),
    [
        ("rtol", 1e-3, 1e-7),
        ("atol", 1e-3, 0.0),
        ("equal_nan", False, True),
        ("strict_dtype", False, True),
    ],
)
def test_an_explicit_default_in_an_override_still_overrides(name, parent, default):
    tol = ToleranceConfig(
        **{name: parent}, path_overrides={"$.x": ToleranceConfig(**{name: default})}
    )
    assert getattr(tol.for_path("$.x"), name) == default
    assert getattr(tol.for_path("$.other"), name) == parent


def test_an_unset_field_reads_as_its_default_and_is_not_in_repr_or_eq():
    config = ToleranceConfig(atol=1e-3)

    assert (config.rtol, config.equal_nan, config.strict_dtype) == (1e-7, True, True)
    assert config == ToleranceConfig(rtol=1e-7, atol=1e-3)
    assert repr(config) == repr(ToleranceConfig(rtol=1e-7, atol=1e-3))
    assert "unset" not in repr(config)


def test_overrides_chain_three_levels_deep():
    # root sets atol; "$.a" sets rtol and carries its own override for "$.a.b", which
    # sets equal_nan; "$.a.b.c" sets strict_dtype. Each level keeps what it did not set.
    root = ToleranceConfig(
        atol=1e-3,
        path_overrides={
            "$.a": ToleranceConfig(
                rtol=0.1,
                path_overrides={"$.a.b": ToleranceConfig(equal_nan=False)},
            ),
            "$.a.b.c": ToleranceConfig(strict_dtype=False),
        },
    )
    # A frame at "$.a" resolves its columns against the config resolved for the frame,
    # so resolving in steps must agree with resolving straight from the root.
    stepwise = root.for_path("$.a").for_path("$.a.b").for_path("$.a.b.c")
    direct = root.for_path("$.a.b.c")

    for effective in (stepwise, direct):
        assert effective.atol == 1e-3
        assert effective.rtol == 0.1
        assert effective.equal_nan is False
        assert effective.strict_dtype is False
    assert root.for_path("$.a.b") == ToleranceConfig(
        rtol=0.1,
        atol=1e-3,
        equal_nan=False,
        path_overrides=root.for_path("$.a.b").path_overrides,
    )


def test_a_shorter_override_never_undoes_a_longer_one():
    # "$.a.b" brings an override for "$.a" of its own: shorter, so it must not win.
    root = ToleranceConfig(
        path_overrides={
            "$.a": ToleranceConfig(atol=1.0),
            "$.a.b": ToleranceConfig(
                atol=2.0, path_overrides={"$.a": ToleranceConfig(atol=9.0)}
            ),
        }
    )
    assert root.for_path("$.a.b.c").atol == 2.0


@pytest.mark.parametrize(
    ("expected", "actual", "root"),
    [
        ({"x": 0.0, "y": 0.0}, {"x": 5e-4, "y": 5e-4}, "$.x"),
        (np.zeros(3), np.full(3, 5e-4), "$"),
    ],
    ids=["dict", "numpy"],
)
def test_an_rtol_only_override_keeps_the_parents_atol_in_compare(
    expected, actual, root
):
    # Zero expected values: rtol alone can never pass them, only the inherited atol can.
    tol = ToleranceConfig(atol=1e-3, path_overrides={root: ToleranceConfig(rtol=0.1)})

    assert compare(expected, actual, tol)
    assert not compare(
        expected,
        actual,
        ToleranceConfig(path_overrides={root: ToleranceConfig(rtol=0.1)}),
    )


def test_a_replaced_config_counts_every_field_as_given():
    # replace() passes every field to the constructor, so as an override it sets them
    # all - including the defaults it merely copied.
    derived = dataclasses.replace(ToleranceConfig(), rtol=0.1)
    tol = ToleranceConfig(atol=1e-3, path_overrides={"$.x": derived})

    assert tol.for_path("$.x").rtol == 0.1
    assert tol.for_path("$.x").atol == 0.0
    # The constructor with only rtol inherits instead.
    tol = ToleranceConfig(atol=1e-3, path_overrides={"$.x": ToleranceConfig(rtol=0.1)})
    assert tol.for_path("$.x").atol == 1e-3


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


def test_a_registered_comparator_goes_after_the_built_ins_and_before_objects():
    class ClaimsEverything:
        def can_handle(self, expected, actual):
            return True

        def compare(self, expected, actual, path, tolerance, compare_fn, diffs):
            diffs.append(Diff(path, "value_mismatch", "", "", "custom"))

    specific, custom = NumpyArrayComparator(), ClaimsEverything()
    registry = ComparatorRegistry()
    registry.register(specific)
    registry.register(custom)

    assert registry.find(np.zeros(2), np.zeros(2)) is specific
    # Consulted before the attribute walk, so it takes over an object it claims.
    result = compare(Point(1, 2), Point(1, 2), registry=registry)
    assert [d.message for d in result.diffs] == ["custom"]


def test_registering_on_a_copy_leaves_the_original_alone():
    class ClaimsEverything:
        def can_handle(self, expected, actual):
            return True

        def compare(
            self, expected, actual, path, tolerance, compare_fn, diffs
        ):  # pragma: no cover
            pass

    registry = ComparatorRegistry()
    registry.register(NumpyArrayComparator())

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


# ------------------------------------------------------- review regression tests


def test_integers_too_large_for_a_float_compare_exactly_and_never_raise():
    """``float(10**400)`` raised OverflowError out of a function that never raises."""
    assert compare(10**400, 10**400)
    assert kinds(compare(10**400, 10**400 + 1)) == ["value_mismatch"]
    # Above 2**53 a float cannot separate neighbours, so these used to compare equal.
    assert kinds(compare(2**60, 2**60 + 1)) == ["value_mismatch"]


def test_a_value_whose_repr_raises_does_not_escape_compare():
    class Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")

        def __eq__(self, other):
            return False

        __hash__ = None

    result = compare({"a": 1}, {"a": 1, "extra": Hostile()})
    assert kinds(result) == ["extra_key"]
    assert "repr() raised RuntimeError" in result.diffs[0].actual


class _Pair(tuple):
    """A tuple subclass that is not a named tuple: recorded as a plain tuple."""


class _Items(list):
    """A list subclass: recorded as a plain list."""


def test_builtin_sequence_subclasses_compare_as_the_builtin_they_are_recorded_as():
    """Comparing exact types made every record of such a value fail forever."""
    assert compare((1, 2), _Pair((1, 2)))
    assert compare([1, 2], _Items([1, 2]))
    assert kinds(compare([1, 2], _Pair((1, 2)))) == ["type_mismatch"]


def test_a_named_tuple_is_its_own_kind_of_sequence():
    from collections import namedtuple

    Point = namedtuple("Point", "x y")
    assert compare(Point(1, 2), Point(1, 2))
    assert kinds(compare(Point(1, 2), (1, 2))) == ["type_mismatch"]
    assert kinds(compare(Point(1, 2), Point(1, 3))) == ["value_mismatch"]


def test_a_column_override_narrows_a_broader_override_on_the_frame():
    """The case docs/comparison.md promises: the broad override forgot the narrow one."""
    expected = pl.DataFrame({"price": [100.0], "qty": [10.0]})
    actual = pl.DataFrame({"price": [100.5], "qty": [10.5]})
    tolerance = ToleranceConfig(
        rtol=0.0,
        path_overrides={
            "$": ToleranceConfig(rtol=0.0, atol=1.0),
            "$['price']": ToleranceConfig(rtol=0.0, atol=0.0),
        },
    )
    assert paths(compare(expected, actual, tolerance)) == ["$['price']"]


@pytest.mark.parametrize(
    ("dtype", "expected", "actual"),
    [
        ("boolean", [True, None, False], [True, None, True]),
        ("string", ["a", None, "c"], ["a", None, "x"]),
    ],
)
def test_pandas_nullable_column_with_a_missing_value_does_not_raise(
    dtype, expected, actual
):
    """``==`` on pd.NA is NA, and ``.any()`` then raised out of compare()."""
    before = pd.DataFrame({"x": pd.array(expected, dtype=dtype)})
    after = pd.DataFrame({"x": pd.array(actual, dtype=dtype)})
    assert compare(before, before.copy())
    result = compare(before, after)
    assert kinds(result) == ["value_mismatch"]
    assert "position(s): 2" in result.diffs[0].message


@pytest.mark.parametrize(("dtype", "tolerant"), [("Int64", False), ("Float64", True)])
def test_pandas_nullable_columns_floats_within_tolerance_integers_exactly(
    dtype, tolerant
):
    expected = pd.DataFrame({"x": pd.array([1, 2, None], dtype=dtype)})
    drifted = pd.DataFrame({"x": pd.array([1, 3, None], dtype=dtype)})
    gained_na = pd.DataFrame({"x": pd.array([1, None, None], dtype=dtype)})
    assert not compare(expected, drifted)
    assert compare(expected, drifted, ToleranceConfig(atol=1.0)).passed is tolerant
    assert compare(expected, expected.copy())
    assert kinds(compare(expected, gained_na, ToleranceConfig(atol=9.0))) == [
        "value_mismatch"
    ]


def test_pandas_boolean_columns_are_never_within_tolerance():
    expected = pd.DataFrame({"flag": [True, False]})
    actual = pd.DataFrame({"flag": [True, True]})
    assert kinds(compare(expected, actual, ToleranceConfig(atol=2.0))) == [
        "value_mismatch"
    ]


@pytest.mark.parametrize("frame", [pd.DataFrame, pl.DataFrame])
def test_strict_dtype_false_relaxes_dataframe_schemas_too(frame):
    """It already did for arrays, numpy scalars and Series - but not for frames."""
    expected = frame({"x": np.array([1.0, 2.0], dtype=np.float64)})
    actual = frame({"x": np.array([1.0, 2.0], dtype=np.float32)})
    assert kinds(compare(expected, actual)) == ["schema_mismatch"]
    assert compare(expected, actual, ToleranceConfig(strict_dtype=False))


def _categorical_frame(categories, ordered=False):
    return pd.DataFrame(
        {"c": pd.Categorical(["a", "b"], categories=categories, ordered=ordered)}
    )


@pytest.mark.parametrize(
    ("expected", "actual", "passes"),
    [
        ((["a", "b"], False), (["a", "b", "z"], False), False),  # category added
        ((["a", "b"], False), (["a", "b"], True), False),  # became ordered
        ((["a", "b"], True), (["b", "a"], True), False),  # ordering changed
        ((["a", "b"], False), (["b", "a"], False), True),  # unordered: order is moot
    ],
    ids=["added", "ordered", "reordered", "unordered-reorder"],
)
def test_a_categorical_columns_category_set_is_part_of_its_schema(
    expected, actual, passes
):
    result = compare(_categorical_frame(*expected), _categorical_frame(*actual))

    assert result.passed is passes
    if not passes:
        assert result.diffs[0].kind == "schema_mismatch"


def test_max_abs_diff_of_complex_arrays_keeps_the_imaginary_part():
    import warnings

    expected = np.array([1 + 1j, 2 + 0j])
    actual = np.array([1 + 1j, 2 + 3j])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a ComplexWarning would fail the test
        result = compare(expected, actual)

    assert result.diffs[0].tolerance_used.max_abs_diff == pytest.approx(3.0)


def test_object_and_str_columns_share_a_dtype_but_values_are_compared():
    try:
        text = pd.StringDtype(na_value=np.nan)
    except TypeError:
        pytest.skip("this pandas has no NaN-missing string dtype")
    as_object = pd.DataFrame({"s": pd.Series(["a", "b"], dtype=object)})

    assert compare(as_object, as_object.astype(text)).passed
    changed = compare(as_object, pd.DataFrame({"s": pd.Series(["a", "x"], dtype=text)}))
    assert changed.diffs[0].kind == "value_mismatch"
    # The opt-in pd.NA-missing "string" dtype is a different dtype.
    assert not compare(as_object, as_object.astype("string")).passed


# --- integers compare exactly (#9) --------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (1_000_000_000, 1_000_000_050),
        (np.int64(7), np.int64(8)),
        (np.array([1, 2]), np.array([1, 3])),
        (np.array([1, 2], dtype=np.uint8), np.array([1, 3], dtype=np.uint8)),
        (pd.DataFrame({"n": [1, 2]}), pd.DataFrame({"n": [1, 3]})),
        (pl.DataFrame({"n": [1, 2]}), pl.DataFrame({"n": [1, 3]})),
    ],
    ids=["int", "numpy-scalar", "array", "uint8-array", "pandas", "polars"],
)
def test_integers_compare_exactly_whatever_the_tolerance(expected, actual):
    loose = ToleranceConfig(rtol=0.5, atol=10)

    result = compare(expected, actual, loose)

    assert kinds(result) == ["value_mismatch"]
    assert result.diffs[0].tolerance_used is None


def test_an_integer_against_a_float_is_still_compared_within_tolerance():
    assert compare(1_000_000_000, 1_000_000_050.0, ToleranceConfig(rtol=1e-6))
    loose_dtype = ToleranceConfig(strict_dtype=False)
    assert compare(np.array([1, 2]), np.array([1.0, 2.0000001]), loose_dtype)
