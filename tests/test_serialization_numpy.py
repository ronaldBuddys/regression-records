import numpy as np
import pytest

from regression_records.exceptions import SerializationError
from regression_records.serialization import to_json_tree

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
