"""Types beyond JSON's own that must load back as the type they were recorded as."""

import datetime
import decimal
import enum
import json
import math
import pathlib

import pandas as pd
import pytest

from regression_records.comparison import compare
from regression_records.exceptions import IdentityResolutionError, SerializationError
from regression_records.serialization import from_json_tree, to_json_tree

from ._helpers import round_trip


class Colour(enum.Enum):
    RED = "r"
    GREEN = (1, 2)


class Perm(enum.Flag):
    R = 4
    W = 2


NEW_YORK = "America/New_York"


def _values():
    from zoneinfo import ZoneInfo

    return {
        "frozenset": frozenset({3, "a", (1, 2)}),
        "bytearray": bytearray(b"\x00\xffab"),
        "time": datetime.time(1, 2, 3, 456789),
        "time-aware": datetime.time(1, 2, tzinfo=datetime.UTC),
        "time-fold": datetime.time(1, 30, fold=1),
        "timedelta": datetime.timedelta(days=-2, seconds=5, microseconds=7),
        "decimal": decimal.Decimal("1.10"),
        "decimal-nan": decimal.Decimal("NaN"),
        "decimal-neg-zero": decimal.Decimal("-0.000"),
        "path": pathlib.Path("data") / "file.csv",
        "pure-windows-path": pathlib.PureWindowsPath(r"C:\data\file.csv"),
        # A path value to round-trip, never opened.
        "pure-posix-path": pathlib.PurePosixPath("/tmp/x"),  # noqa: S108
        "complex": complex(1.5, -2.0),
        "complex-nan": complex(math.nan, math.inf),
        "enum": Colour.GREEN,
        "flag": Perm.W,
        "timestamp-ns": pd.Timestamp("2026-01-01 00:00:00.000000001"),
        "timestamp-zone": pd.Timestamp("2026-03-01 09:30:00.000000005", tz=NEW_YORK),
        "timestamp-utc": pd.Timestamp("2026-01-01", tz="UTC"),
        "timestamp-zoneinfo": pd.Timestamp("2026-07-01 12:00", tz=ZoneInfo(NEW_YORK)),
        "nat": pd.NaT,
        "pandas-timedelta": pd.Timedelta(1, unit="ns") + pd.Timedelta(days=1),
    }


VALUES = _values()


@pytest.mark.parametrize("name", VALUES)
def test_a_value_loads_back_as_its_own_type(ctx_pair, name):
    value = VALUES[name]

    tree = to_json_tree(value, ctx_pair[0])
    json.dumps(tree, allow_nan=False)  # strict JSON, as a record is written
    back = round_trip(value, ctx_pair)

    assert type(back) is type(value)
    assert compare(value, back).passed
    if not isinstance(value, frozenset):  # a set's str depends on iteration order
        assert str(back) == str(value)  # digits, exponent, zone, path text


def test_a_decimal_keeps_its_exponent(ctx_pair):
    assert round_trip(decimal.Decimal("1.10"), ctx_pair).as_tuple().exponent == -2


def test_a_zoned_timestamp_keeps_its_zone_through_an_ambiguous_hour(ctx_pair):
    # False picks the second 01:30 (standard time) of the repeated hour.
    value = pd.Timestamp("2026-11-01 01:30").tz_localize(NEW_YORK, ambiguous=False)
    back = round_trip(value, ctx_pair)

    assert str(back.tz) == NEW_YORK
    assert back.utcoffset() == value.utcoffset()


def test_a_composite_flag_is_refused(ctx_pair):
    with pytest.raises(SerializationError, match="single named member"):
        to_json_tree(Perm.R | Perm.W, ctx_pair[0])


def test_an_enum_member_that_no_longer_exists_is_an_identity_error(ctx_pair):
    tree = to_json_tree(Colour.RED, ctx_pair[0])
    tree["member"] = "BLUE"

    with pytest.raises(IdentityResolutionError, match="no member 'BLUE'"):
        from_json_tree(tree, ctx_pair[1])


def test_a_concrete_path_loads_as_this_platforms_path(ctx_pair):
    tree = {"__type__": "pathlib.Path", "value": "a/b", "class": "WindowsPath"}

    assert type(from_json_tree(tree, ctx_pair[1])) is type(pathlib.Path())


def test_a_path_subclass_is_left_to_the_generic_serializer(ctx_pair):
    class MyPath(type(pathlib.Path())):
        pass

    assert to_json_tree(pathlib.Path("x"), ctx_pair[0])["__type__"] == "pathlib.Path"
    with pytest.raises(SerializationError):  # a class inside a function: refused
        to_json_tree(MyPath("x"), ctx_pair[0])


@pytest.mark.parametrize(
    ("expected", "actual", "passes"),
    [
        (decimal.Decimal("NaN"), decimal.Decimal("NaN"), True),
        (complex(math.nan, 1), complex(math.nan, 1), True),
        (complex(math.nan, 1), complex(math.nan, 2), False),  # other part differs
        (pd.NaT, pd.NaT, True),
        (decimal.Decimal("NaN"), decimal.Decimal("1"), False),
    ],
)
def test_nan_like_values_follow_equal_nan(expected, actual, passes):
    from regression_records.comparison import ToleranceConfig

    assert compare(expected, actual).passed is passes
    assert not compare(expected, actual, ToleranceConfig(equal_nan=False)).passed


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (decimal.Decimal("1.1"), decimal.Decimal("1.2")),
        (pathlib.PurePosixPath("a"), pathlib.PurePosixPath("b")),
        (Colour.RED, Colour.GREEN),
    ],
)
def test_value_types_are_compared_by_equality(expected, actual):
    result = compare(expected, actual)

    assert result.diffs[0].kind == "value_mismatch"
    assert repr(actual) in result.diffs[0].message
