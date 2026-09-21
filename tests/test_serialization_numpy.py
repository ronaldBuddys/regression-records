import hashlib

import numpy as np
import pytest

from regression_records.exceptions import SerializationError
from regression_records.serialization import to_json_tree
from regression_records.serialization.naming import content_filename
from regression_records.serialization.numpy_serializer import _raw_bytes

from ._helpers import round_trip


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1.0, 2.5, np.nan], dtype=np.float64),
        np.array([True, False, True]),
        np.zeros((3, 4)),
        np.arange(24).reshape(2, 3, 4),
        np.array([], dtype=np.float64),
        np.arange(5).astype("datetime64[D]"),
        np.array(["a", "bb", "ccc"]),
    ],
)
def test_array_round_trip(arr, ctx_pair):
    result = round_trip(arr, ctx_pair)
    assert result.dtype == arr.dtype
    assert result.shape == arr.shape
    np.testing.assert_array_equal(result, arr, strict=False)


def test_array_nan_round_trip_exact(ctx_pair):
    arr = np.array([1.0, np.nan, 3.0])
    result = round_trip(arr, ctx_pair)
    assert np.isnan(result[1])


@pytest.mark.parametrize(
    "scalar",
    [
        np.int64(42),
        np.float64(3.14),
        np.bool_(True),
        np.datetime64("2024-03-14"),
    ],
)
def test_scalar_round_trip(scalar, ctx_pair):
    result = round_trip(scalar, ctx_pair)
    assert isinstance(result, np.generic)
    assert result.dtype == scalar.dtype
    assert result == scalar


def test_object_dtype_array_rejected(ctx_pair):
    ser_ctx, _ = ctx_pair
    arr = np.array([1, "two", None], dtype=object)
    with pytest.raises(SerializationError, match="dtype=object"):
        to_json_tree(arr, ser_ctx)


def test_content_hash_dedup(ctx_pair, tmp_path):
    ser_ctx, _ = ctx_pair
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    tree1 = to_json_tree(arr1, ser_ctx)
    tree2 = to_json_tree(arr2, ser_ctx)
    assert tree1["path"] == tree2["path"]
    npy_files = list(tmp_path.glob("*.npy"))
    assert len(npy_files) == 1


def test_different_content_different_files(ctx_pair, tmp_path):
    ser_ctx, _ = ctx_pair
    to_json_tree(np.array([1, 2, 3]), ser_ctx)
    to_json_tree(np.array([1, 2, 4]), ser_ctx)
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_shape_mismatch_produces_different_hash(ctx_pair):
    ser_ctx, _ = ctx_pair
    tree1 = to_json_tree(np.array([1, 2, 3, 4]).reshape(2, 2), ser_ctx)
    tree2 = to_json_tree(np.array([1, 2, 3, 4]).reshape(4, 1), ser_ctx)
    assert tree1["path"] != tree2["path"]


def test_nested_arrays_round_trip(ctx_pair):
    value = {
        "x": np.array([1, 2, 3]),
        "y": [np.array([4.0, 5.0]), np.array([6.0])],
        "meta": {"count": 2},
    }
    result = round_trip(value, ctx_pair)
    np.testing.assert_array_equal(result["x"], value["x"])
    np.testing.assert_array_equal(result["y"][0], value["y"][0])
    assert result["meta"] == {"count": 2}


# ------------------------------------------------------------------ content hashing


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(6.0),
        np.arange(6).reshape(2, 3),
        np.asfortranarray(np.arange(6).reshape(2, 3)),
        np.arange(12)[::2],
        np.array(3.0),
        np.arange(5).astype("datetime64[D]"),
        np.arange(3).astype("timedelta64[s]"),
        np.array(["a", "bb", "ccc"]),
        np.array([], dtype=np.float64),
        np.array([1 + 2j]),
    ],
)
def test_array_content_hash_matches_its_own_bytes(arr):
    """The digest is hashed from a buffer view rather than a copy of the array.

    It must stay byte-for-byte what ``tobytes()`` would have produced - the filename is
    the content hash, so a change here would silently rewrite every side file in every
    existing store under a new name, orphaning the old ones.
    """
    digest = hashlib.sha256()
    for chunk in (arr.tobytes(), str(arr.shape).encode(), str(arr.dtype).encode()):
        digest.update(len(chunk).to_bytes(8, "little") + chunk)
    expected = digest.hexdigest()[:32]
    assert (
        content_filename(
            "arr",
            _raw_bytes(arr),
            str(arr.shape).encode(),
            str(arr.dtype).encode(),
            extension="npy",
        )
        == f"arr-{expected}.npy"
    )


def test_a_non_contiguous_array_hashes_as_its_contiguous_equal(ctx_pair):
    ser_ctx, _ = ctx_pair
    strided = np.arange(12)[::2]
    assert (
        to_json_tree(strided, ser_ctx)["path"]
        == to_json_tree(np.ascontiguousarray(strided), ser_ctx)["path"]
    )


# ------------------------------------------------------- review regression tests


def test_masked_array_is_refused_rather_than_losing_its_mask(ctx_pair):
    """Two arrays differing only in mask shared a content hash - and one input hash."""
    masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
    with pytest.raises(SerializationError, match="MaskedArray's mask"):
        to_json_tree(masked, ctx_pair[0])
    assert round_trip({"data": masked.data, "mask": masked.mask}, ctx_pair)


def test_structured_array_with_an_object_field_is_refused_cleanly(ctx_pair):
    arr = np.array([(1, "a")], dtype=[("n", "i4"), ("label", "O")])
    with pytest.raises(SerializationError, match="dtype=object"):
        to_json_tree(arr, ctx_pair[0])


def test_bytes_scalar_round_trips(ctx_pair):
    """It was stored as its repr, and came back as the five characters of "b'ab'"."""
    scalar = np.array([b"ab"])[0]
    result = round_trip(scalar, ctx_pair)
    assert result == scalar
    assert type(result) is np.bytes_


def test_timedelta_too_long_for_nanoseconds_is_refused(ctx_pair):
    """Scalars are stored as nanoseconds, which count ~292 years at most."""
    with pytest.raises(SerializationError, match="does not fit in the nanoseconds"):
        to_json_tree(np.timedelta64(400_000, "D"), ctx_pair[0])


@pytest.mark.parametrize("unit", ["Y", "M", "D", "s"])
def test_timedelta_units_round_trip(ctx_pair, unit):
    value = np.timedelta64(3, unit)
    result = round_trip(value, ctx_pair)
    assert result == value
    assert result.dtype == value.dtype


def test_timedelta_nat_still_round_trips(ctx_pair):
    assert np.isnat(round_trip(np.timedelta64("NaT", "s"), ctx_pair))
