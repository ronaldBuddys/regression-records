import json
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from regression_records.exceptions import CorruptRecordError
from regression_records.identity import FunctionIdentity
from regression_records.records import (
    ExceptionRecord,
    Invocation,
    RegressionRecord,
    SerializedValue,
)
from regression_records.serialization import to_json_tree
from regression_records.store import RecordStore


def sample_function(a, b, scale=1):
    return (a + b) * scale


def _make_record(store, input_hash="hash1", output_obj=None, tags=None):
    identity = FunctionIdentity.from_function(sample_function)
    ser_ctx = store.serialization_context(identity)
    invocation = Invocation(
        args=SerializedValue(to_json_tree([1, 2], ser_ctx)),
        kwargs=SerializedValue(to_json_tree({"scale": 2}, ser_ctx)),
    )
    output = SerializedValue(to_json_tree(output_obj, ser_ctx))
    return RegressionRecord.new(
        identity=identity,
        input_hash=input_hash,
        invocation=invocation,
        output=output,
        tags=tags,
    )


def test_write_and_load_round_trip(tmp_path):
    store = RecordStore(tmp_path)
    record = _make_record(store, output_obj=6)
    store.save(record)

    loaded = store.list_records(record.identity)
    assert len(loaded) == 1
    assert loaded[0].record_id == record.record_id
    assert loaded[0].input_hash == "hash1"

    deser_ctx = store.deserialization_context(record.identity)
    assert loaded[0].load_output(deser_ctx) == 6
    assert loaded[0].load_args(deser_ctx) == [1, 2]
    assert loaded[0].load_kwargs(deser_ctx) == {"scale": 2}


def test_none_output_distinguished_from_no_output(tmp_path):
    store = RecordStore(tmp_path)
    identity = FunctionIdentity.from_function(sample_function)
    ser_ctx = store.serialization_context(identity)
    invocation = Invocation(
        args=SerializedValue(to_json_tree([], ser_ctx)), kwargs=SerializedValue({})
    )
    record_with_none = RegressionRecord.new(
        identity=identity,
        input_hash="h-none",
        invocation=invocation,
        output=SerializedValue(to_json_tree(None, ser_ctx)),
    )
    record_with_exception = RegressionRecord.new(
        identity=identity,
        input_hash="h-exc",
        invocation=invocation,
        exception=ExceptionRecord(
            exc_type="ValueError", message="boom", traceback_text="..."
        ),
    )
    store.save(record_with_none)
    store.save(record_with_exception)

    reloaded = {r.input_hash: r for r in store.list_records(identity)}
    assert reloaded["h-none"].output is not None
    assert reloaded["h-none"].succeeded
    assert (
        reloaded["h-none"].load_output(store.deserialization_context(identity)) is None
    )

    assert reloaded["h-exc"].output is None
    assert not reloaded["h-exc"].succeeded
    assert reloaded["h-exc"].exception.exc_type == "ValueError"


def test_external_file_paths_are_relative(tmp_path):
    store = RecordStore(tmp_path)
    record = _make_record(store, output_obj=np.array([1, 2, 3]))
    path = store.save(record)

    with open(path) as f:
        raw = json.load(f)
    file_path = raw["output"]["path"]
    assert not file_path.startswith("/")
    assert ".." in file_path or "data" in file_path


def test_multiple_records_per_identity(tmp_path):
    store = RecordStore(tmp_path)
    r1 = _make_record(store, input_hash="hash1", output_obj=1)
    r2 = _make_record(store, input_hash="hash2", output_obj=2)
    store.save(r1)
    store.save(r2)

    records = store.list_records(r1.identity)
    assert len(records) == 2
    assert {r.input_hash for r in records} == {"hash1", "hash2"}

    found = store.find_by_input_hash(r1.identity, "hash2")
    assert found is not None
    assert found.record_id == r2.record_id
    assert store.find_by_input_hash(r1.identity, "does-not-exist") is None


def test_save_without_overwrite_rejects_duplicate_id(tmp_path):
    store = RecordStore(tmp_path)
    record = _make_record(store, output_obj=1)
    store.save(record)
    with pytest.raises(FileExistsError):
        store.save(record)
    store.save(record, overwrite=True)  # should not raise


def test_corrupt_json_raises(tmp_path):
    store = RecordStore(tmp_path)
    record = _make_record(store, output_obj=1)
    path = store.save(record)
    path.write_text("{not valid json")

    with pytest.raises(CorruptRecordError):
        store.list_records(record.identity)


def test_malformed_record_raises(tmp_path):
    store = RecordStore(tmp_path)
    record = _make_record(store, output_obj=1)
    path = store.save(record)
    path.write_text(json.dumps({"record_id": "abc"}))  # missing required fields

    with pytest.raises(CorruptRecordError):
        store.list_records(record.identity)


def test_missing_referenced_file_raises(tmp_path):
    store = RecordStore(tmp_path)
    record = _make_record(store, output_obj=np.array([1, 2, 3]))
    store.save(record)

    data_dir = store._data_dir(record.identity)
    for npy_file in data_dir.glob("*.npy"):
        npy_file.unlink()

    loaded = store.list_records(record.identity)[0]
    with pytest.raises(CorruptRecordError):
        loaded.load_output(store.deserialization_context(record.identity))


def test_store_directory_can_be_moved(tmp_path):
    original_root = tmp_path / "original"
    original_root.mkdir()
    store = RecordStore(original_root)
    record = _make_record(store, output_obj=np.array([4, 5, 6]))
    store.save(record)

    moved_root = tmp_path / "moved"
    shutil.move(str(original_root), str(moved_root))

    new_store = RecordStore(moved_root)
    loaded = new_store.list_records(record.identity)[0]
    output = loaded.load_output(new_store.deserialization_context(record.identity))
    np.testing.assert_array_equal(output, np.array([4, 5, 6]))


def test_fresh_process_round_trip(tmp_path, request):
    # Import the SAME sample_function (not a redefinition) so both processes derive
    # the identical FunctionIdentity - a function defined inline in a `-c` script
    # would have __module__ == "__main__", a different identity entirely.
    repo_root = request.config.rootpath
    script = textwrap.dedent(
        f"""
        from tests.test_records_persistence import sample_function
        from regression_records.identity import FunctionIdentity
        from regression_records.records import Invocation, RegressionRecord, SerializedValue
        from regression_records.serialization import to_json_tree
        from regression_records.store import RecordStore

        store = RecordStore({str(tmp_path)!r})
        identity = FunctionIdentity.from_function(sample_function)
        ser_ctx = store.serialization_context(identity)
        invocation = Invocation(
            args=SerializedValue(to_json_tree([1, 2], ser_ctx)),
            kwargs=SerializedValue(to_json_tree({{}}, ser_ctx)),
        )
        output = SerializedValue(to_json_tree(6, ser_ctx))
        record = RegressionRecord.new(
            identity=identity, input_hash="cross-process", invocation=invocation, output=output
        )
        store.save(record)
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    store = RecordStore(tmp_path)
    identity = FunctionIdentity.from_function(sample_function)
    record = store.find_by_input_hash(identity, "cross-process")
    assert record is not None
    assert record.load_output(store.deserialization_context(identity)) == 6


# ------------------------------------------------------------- moving an identity


def other_function(a, b, scale=1):
    """A stand-in for ``sample_function`` after a rename or a move between modules."""
    return (a + b) * scale


def test_rename_identity_moves_records_and_their_data(tmp_path):
    store = RecordStore(tmp_path)
    old = FunctionIdentity.from_function(sample_function)
    new = FunctionIdentity.from_function(other_function)
    store.save(_make_record(store, input_hash="moved", output_obj=np.arange(4)))

    store.rename_identity(old, new)

    assert not store._identity_dir(old).exists()
    record = store.find_by_input_hash(new, "moved")
    assert record is not None
    # Side-file references are relative, so the data moves with the directory and keeps
    # resolving without anything being rewritten.
    loaded = record.load_output(store.deserialization_context(new))
    assert np.array_equal(loaded, np.arange(4))


def test_rename_identity_updates_where_each_record_says_it_lives(tmp_path):
    store = RecordStore(tmp_path)
    old = FunctionIdentity.from_function(sample_function)
    new = FunctionIdentity.from_function(other_function)
    store.save(_make_record(store, input_hash="moved", output_obj=6))

    store.rename_identity(old, new)
    record = store.find_by_input_hash(new, "moved")

    assert (record.identity.module, record.identity.qualname) == (
        new.module,
        new.qualname,
    )
    # ... but what it was captured *from* is history, and survives the move - otherwise
    # moving a function would silently erase its code-drift signal.
    assert record.identity.code_hash == old.code_hash


def test_rename_identity_refuses_when_there_is_nothing_to_move(tmp_path):
    store = RecordStore(tmp_path)
    old = FunctionIdentity.from_function(sample_function)
    new = FunctionIdentity.from_function(other_function)
    with pytest.raises(FileNotFoundError, match="no records exist"):
        store.rename_identity(old, new)


def test_rename_identity_refuses_to_clobber_an_existing_history(tmp_path):
    store = RecordStore(tmp_path)
    old = FunctionIdentity.from_function(sample_function)
    new = FunctionIdentity.from_function(other_function)
    store.save(_make_record(store, input_hash="moved", output_obj=6))
    store._records_dir(new).mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exist"):
        store.rename_identity(old, new)


def test_a_record_from_a_newer_schema_is_refused(tmp_path):
    """Reading a future format as this one risks a wrong answer, not an error."""
    store = RecordStore(tmp_path)
    path = store.save(_make_record(store, output_obj=1))
    raw = json.loads(path.read_text())
    raw["schema"]["schema_version"] += 1
    path.write_text(json.dumps(raw))

    identity = FunctionIdentity.from_function(sample_function)
    with pytest.raises(CorruptRecordError, match="upgrade regression-records"):
        store.list_records(identity)


def test_saving_leaves_no_temporary_files_behind(tmp_path):
    store = RecordStore(tmp_path)
    path = store.save(_make_record(store, output_obj=1))
    assert [p.name for p in path.parent.iterdir()] == [path.name]
