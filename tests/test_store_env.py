"""The store directory from ``REGRESSION_RECORDS_STORE`` or a ``.env`` file."""

import pytest

from regression_records import (
    RecordStore,
    StoreNotConfiguredError,
    record_func,
    record_method,
    replay,
)
from regression_records.store import STORE_ENV, coerce_store

from . import _replay_targets as targets


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Start with no variable, in an empty directory with no ``.env`` above it."""
    monkeypatch.delenv(STORE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)


def test_explicit_store_beats_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "from_env"))
    store = coerce_store(tmp_path / "explicit")
    assert store.root == tmp_path / "explicit"


def test_a_record_store_is_returned_as_is(monkeypatch, tmp_path):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "from_env"))
    store = RecordStore(tmp_path / "explicit")
    assert coerce_store(store) is store


def test_environment_variable_names_the_store(monkeypatch, tmp_path):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "from_env"))
    assert coerce_store().root == tmp_path / "from_env"


def test_tilde_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(STORE_ENV, "~/records")
    assert coerce_store().root == tmp_path / "records"


def test_dotenv_in_a_parent_directory_is_found(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"{STORE_ENV}=from_dotenv\n")
    work = tmp_path / "sub" / "dir"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    assert coerce_store().root.name == "from_dotenv"


def test_real_environment_beats_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"{STORE_ENV}=from_dotenv\n")
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "from_env"))
    assert coerce_store().root == tmp_path / "from_env"


@pytest.mark.parametrize("value", [None, "", "  "])
def test_unset_or_blank_is_a_clear_error(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(STORE_ENV, value)
    with pytest.raises(StoreNotConfiguredError, match=STORE_ENV):
        coerce_store()


def test_not_configured_is_also_a_value_error():
    with pytest.raises(ValueError):
        coerce_store()


def test_record_func_without_a_store_records_to_the_configured_one(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "records"))
    double = record_func()(_double)
    double(2)
    assert list((tmp_path / "records").rglob("rec-*.json"))


def test_record_method_without_a_store_records_to_the_configured_one(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "records"))
    Counter.bump = record_method()(_bump)
    Counter().bump()
    assert list((tmp_path / "records").rglob("rec-*.json"))


def test_decorating_without_a_store_fails_when_unconfigured():
    with pytest.raises(StoreNotConfiguredError):
        record_func()


def test_replay_without_a_store_reads_the_configured_one(monkeypatch, tmp_path):
    root = tmp_path / "records"
    monkeypatch.setenv(STORE_ENV, str(root))
    record_func(root)(targets.total.__wrapped__)([1.0, 2.0])
    assert replay().passed


def test_replay_without_a_store_is_an_error_when_unconfigured():
    with pytest.raises(StoreNotConfiguredError):
        replay()


def _double(x):
    return 2 * x


class Counter:
    def __init__(self):
        self.n = 0


def _bump(self):
    self.n += 1
    return self.n
