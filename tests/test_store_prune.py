"""RecordStore.prune: data files no record references, listed or deleted."""

import numpy as np
import pytest

from regression_records.identity import FunctionIdentity
from regression_records.records import Invocation, RegressionRecord, SerializedValue
from regression_records.serialization import to_json_tree
from regression_records.store import RecordStore


def _identity(module="pkg.mod", qualname="f"):
    return FunctionIdentity(
        module=module, qualname=qualname, kind="function", code_hash="", signature=""
    )


def _save(store, identity, output):
    """Save a record whose output is ``output``; return the record's path."""
    ctx = store.serialization_context(identity)
    record = RegressionRecord.new(
        identity=identity,
        input_hash="h" * 64,
        invocation=Invocation(args=SerializedValue([]), kwargs=SerializedValue({})),
        output=SerializedValue(to_json_tree(output, ctx)),
    )
    return store.save(record)


def _data_files(store, identity):
    return sorted((store._identity_dir(identity) / "data").iterdir())


@pytest.fixture()
def store(tmp_path):
    return RecordStore(tmp_path / "store")


def test_only_files_whose_record_was_removed_are_listed(store):
    identity = _identity()
    kept = _save(store, identity, np.arange(3))
    removed = _save(store, identity, np.arange(4))
    [kept_file, orphan] = sorted(
        _data_files(store, identity),
        key=lambda p: p.name not in kept.read_text(),
    )
    removed.unlink()  # removing a record is a manual act

    assert store.prune(identity) == [orphan]
    assert orphan.exists()  # a dry run by default

    assert store.prune(identity, dry_run=False) == [orphan]
    assert not orphan.exists()
    assert kept_file.exists()
    assert store.prune(identity) == []


def test_the_whole_store_is_pruned_when_no_identity_is_given(store):
    # A module named "data" makes a directory of that name that is not a data/.
    first, second = _identity("pkg.data", "f"), _identity("other", "g")
    for identity in (first, second):
        _save(store, identity, np.arange(2)).unlink()

    orphans = store.prune()

    assert orphans == sorted(_data_files(store, first) + _data_files(store, second))
    store.prune(dry_run=False)
    assert store.prune() == []
    assert (store.root / "pkg" / "data").is_dir()  # the module directory is untouched


def test_a_record_that_no_longer_parses_still_protects_its_files(store):
    identity = _identity()
    path = _save(store, identity, np.arange(3))
    path.write_text(path.read_text()[:-20])  # truncated: invalid JSON

    assert store.prune(identity) == []


def test_files_left_by_an_interrupted_save_are_listed(store):
    identity = _identity()
    ctx = store.serialization_context(identity)
    to_json_tree(np.arange(5), ctx)  # side file written, record never saved

    assert store.prune(identity) == _data_files(store, identity)


def test_a_missing_store_or_identity_has_nothing_to_prune(tmp_path):
    assert RecordStore(tmp_path / "nowhere").prune() == []
    assert RecordStore(tmp_path).prune(_identity()) == []
