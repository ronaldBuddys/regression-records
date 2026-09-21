"""Side-file names: content hash length, chunk framing, legacy names and verify-on-read."""

import re
import shutil

import numpy as np
import pytest

from regression_records.exceptions import CorruptRecordError
from regression_records.identity import FunctionIdentity
from regression_records.serialization import from_json_tree, to_json_tree
from regression_records.serialization.naming import (
    CONTENT_HASH_CHARS,
    SHORT_HASH_CHARS,
    content_hash,
    hash_bytes,
)
from regression_records.serialization.numpy_serializer import _hash_chunks
from regression_records.store import RecordStore, record_filename_prefix


def _numpy(seed):
    return np.arange(6.0) + seed


def _polars(seed):
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"x": [1.0, 2.0, 3.0 + seed]})


def _pandas(seed):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"x": [1.0, 2.0, 3.0 + seed]})


def _legacy_digest(value, path):
    """The short hash a side file was named by before the longer one."""
    if isinstance(value, np.ndarray):
        return hash_bytes(*_hash_chunks(value))
    return hash_bytes(path.read_bytes())


def _equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a.equals(b)


MAKERS = pytest.mark.parametrize(
    "make", [_numpy, _polars, _pandas], ids=["numpy", "polars", "pandas"]
)


def sample_function(x):
    return x


@pytest.fixture()
def store(tmp_path):
    return RecordStore(tmp_path)


@pytest.fixture()
def identity():
    return FunctionIdentity.from_function(sample_function)


def _write(store, identity, value):
    tree = to_json_tree(value, store.serialization_context(identity))
    path = store.deserialization_context(identity).resolve(tree["path"])
    return tree, path


def _load(store, identity, tree, *, verify):
    ctx = store.deserialization_context(identity)
    ctx.verify = verify
    return from_json_tree(tree, ctx)


# --- the hashes themselves ------------------------------------------------------------


def test_content_hash_frames_each_chunk_by_its_length():
    assert content_hash(b"ab", b"c") != content_hash(b"a", b"bc")
    # The short hash concatenates, as side files were named before.
    assert hash_bytes(b"ab", b"c") == hash_bytes(b"a", b"bc")


def test_hash_lengths():
    assert len(content_hash(b"x")) == CONTENT_HASH_CHARS == 32
    assert len(hash_bytes(b"x")) == SHORT_HASH_CHARS == 16


def test_record_filenames_use_the_short_hash():
    assert record_filename_prefix("a" * 64) == f"rec-{'a' * SHORT_HASH_CHARS}-"


@MAKERS
def test_new_side_files_are_named_by_the_long_hash(store, identity, make):
    _, path = _write(store, identity, make(0))

    assert re.fullmatch(rf"[a-z]+-[0-9a-f]{{{CONTENT_HASH_CHARS}}}", path.stem)


# --- legacy names ---------------------------------------------------------------------


@MAKERS
@pytest.mark.parametrize("verify", [False, True])
def test_a_side_file_with_a_legacy_short_name_still_loads(
    store, identity, make, verify
):
    value = make(0)
    tree, path = _write(store, identity, value)
    # Rename it as it would have been written before the longer hash.
    legacy = path.with_name(
        f"{path.stem.rpartition('-')[0]}-{_legacy_digest(value, path)}{path.suffix}"
    )
    path.rename(legacy)
    tree["path"] = tree["path"].replace(path.name, legacy.name)

    assert _equal(_load(store, identity, tree, verify=verify), value)


# --- verify on read -------------------------------------------------------------------


@MAKERS
def test_verification_catches_a_swapped_side_file(store, identity, make):
    recorded, path = _write(store, identity, make(0))
    _, other = _write(store, identity, make(1))
    shutil.copyfile(other, path)

    # Unverified, the swap goes unnoticed: the record loads the other value.
    assert _equal(_load(store, identity, recorded, verify=False), make(1))
    with pytest.raises(CorruptRecordError, match="does not match the content hash"):
        _load(store, identity, recorded, verify=True)


def test_verification_refuses_a_name_without_a_content_hash(store, identity):
    tree, path = _write(store, identity, _numpy(0))
    path.rename(path.with_name("arr-renamed.npy"))
    tree["path"] = tree["path"].replace(path.name, "arr-renamed.npy")

    assert _equal(_load(store, identity, tree, verify=False), _numpy(0))
    with pytest.raises(CorruptRecordError, match="not named by a content hash"):
        _load(store, identity, tree, verify=True)


def test_a_store_hands_out_verifying_contexts_when_asked(tmp_path, identity):
    assert not RecordStore(tmp_path).deserialization_context(identity).verify
    verifying = RecordStore(tmp_path, verify_side_files=True)
    assert verifying.deserialization_context(identity).verify

    tree, path = _write(verifying, identity, _numpy(0))
    shutil.copyfile(_write(verifying, identity, _numpy(1))[1], path)
    with pytest.raises(CorruptRecordError):
        from_json_tree(tree, verifying.deserialization_context(identity))
