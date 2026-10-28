"""Environment variables a recorded call reads: ``env=`` and ``secret_env=``."""

import json
import os

import pytest

from regression_records import RecordStore, ReplayStatus, record_func, record_method
from regression_records import replay as replay_store
from regression_records._env import changed_secrets, fingerprint
from regression_records.records import (
    SCHEMA_VERSION,
    RegressionRecord,
    canonical_payload,
    compute_input_hash,
)

from . import _replay_targets as targets

CURRENCY, API_KEY = targets.CURRENCY, targets.API_KEY


@pytest.fixture()
def store(tmp_path):
    return RecordStore(tmp_path / "records")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(CURRENCY, raising=False)
    monkeypatch.delenv(API_KEY, raising=False)


def record(store, target, *args, **decorator_kwargs):
    """Record one call of ``target`` into ``store``, under ``target``'s identity."""
    return record_func(store, **decorator_kwargs)(target.__wrapped__)(*args)


def record_dicts(store):
    return [json.loads(p.read_text()) for p in store.iter_record_paths()]


def only_case(report):
    assert len(report.cases) == 1
    return report.cases[0]


# ------------------------------------------------------------------ recording


def test_the_named_variables_are_recorded(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")

    assert record(store, targets.in_currency, 5, env=(CURRENCY, "RR_UNSET")) == "5 EUR"

    [data] = record_dicts(store)
    assert data["env"] == {CURRENCY: "EUR", "RR_UNSET": None}
    assert data["schema"]["schema_version"] == SCHEMA_VERSION == 4
    assert "secret_env" not in data


def test_a_record_without_env_has_no_env_key(store):
    record(store, targets.in_currency, 5)

    [data] = record_dicts(store)
    assert "env" not in data and "secret_env" not in data


def test_without_env_the_input_hash_is_unchanged():
    # Stores recorded before env= existed must not re-record every call as new: the
    # hash is pinned to what version 0.4.0 computed.
    args, kwargs = [1, "a"], {"k": 2.5}
    assert canonical_payload(args, kwargs) == '{"args":[1,"a"],"kwargs":{"k":2.5}}'
    assert canonical_payload(args, kwargs, {}) == canonical_payload(args, kwargs)
    assert compute_input_hash(args, kwargs) == (
        "320b01253f2d0a08170861aa3bd99a687877ed59996ea600e5e3ab1e0685bb6b"
    )


def test_another_value_is_another_case_even_with_the_same_result(store, monkeypatch):
    # total ignores the environment, so both calls return the same thing: only the
    # input hash keeps the second recording from coinciding with the first.
    for currency in ("EUR", "USD"):
        monkeypatch.setenv(CURRENCY, currency)
        record(store, targets.total, [1, 2], env=(CURRENCY,))

    records = record_dicts(store)
    assert sorted(r["env"][CURRENCY] for r in records) == ["EUR", "USD"]
    assert len({r["input_hash"] for r in records}) == 2


def test_the_environment_is_read_before_the_call(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")

    def switches_currency(amount):
        os.environ[CURRENCY] = "USD"
        return amount

    switches_currency.__module__ = targets.__name__
    switches_currency.__qualname__ = "switches_currency"
    record_func(store, env=(CURRENCY,))(switches_currency)(5)

    [data] = record_dicts(store)
    assert data["env"] == {CURRENCY: "EUR"}


def test_a_record_written_before_env_existed_loads_with_none():
    data = {
        "schema": {"schema_version": 3, "library_version": "0.3.0"},
        "record_id": "r",
        "input_hash": "h",
        "created_at": "2026-01-01T00:00:00+00:00",
        "identity": {
            "module": "m",
            "qualname": "f",
            "kind": "function",
            "code_hash": "",
            "signature": "",
        },
        "invocation": {"args": [], "kwargs": {}},
        "output": 1,
    }
    record = RegressionRecord.from_dict(data)
    assert record.env == {} and record.secret_env == {}


# ------------------------------------------------------------------ replay


def test_replay_runs_under_the_recorded_value(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")
    record(store, targets.in_currency, 5, env=(CURRENCY,))
    monkeypatch.setenv(CURRENCY, "USD")

    case = only_case(replay_store(store))

    assert case.status is ReplayStatus.PASSED
    assert case.env == {CURRENCY: "EUR"}
    assert os.environ[CURRENCY] == "USD", "put back after the call"


def test_without_env_a_changed_setting_differs(store, monkeypatch):
    # The false alarm env= exists to prevent.
    monkeypatch.setenv(CURRENCY, "EUR")
    record(store, targets.in_currency, 5)
    monkeypatch.setenv(CURRENCY, "USD")

    assert only_case(replay_store(store)).status is ReplayStatus.DIFFERS


def test_a_variable_recorded_as_unset_is_unset_for_the_call(store, monkeypatch):
    record(store, targets.in_currency, 5, env=(CURRENCY,))
    monkeypatch.setenv(CURRENCY, "USD")

    case = only_case(replay_store(store))

    assert case.status is ReplayStatus.PASSED, case.describe()
    assert case.env == {CURRENCY: None}
    assert os.environ[CURRENCY] == "USD"


def test_the_environment_is_put_back_when_the_call_raises(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")
    record(store, targets.in_currency, 5, env=(CURRENCY,))
    monkeypatch.delenv(CURRENCY)

    def boom(amount):
        raise RuntimeError(os.environ[CURRENCY])

    monkeypatch.setattr(targets.in_currency, "__wrapped__", boom)
    case = only_case(replay_store(store))

    assert case.status is ReplayStatus.RAISED
    assert "EUR" in case.error
    assert CURRENCY not in os.environ


def test_a_regression_under_the_recorded_value_is_caught(store, monkeypatch):
    # The other direction: with the setting replayed, a change that only shows under
    # it fails even though the replaying process is set differently.
    monkeypatch.setenv(CURRENCY, "EUR")
    record(store, targets.in_currency, 5, env=(CURRENCY,))
    monkeypatch.setenv(CURRENCY, "USD")

    def broken_for_euros(amount):
        currency = os.environ.get(CURRENCY)
        return f"{amount * (2 if currency == 'EUR' else 1)} {currency}"

    monkeypatch.setattr(targets.in_currency, "__wrapped__", broken_for_euros)

    case = only_case(replay_store(store))
    assert case.status is ReplayStatus.DIFFERS
    assert f"{CURRENCY}='EUR'" in case.describe()


def test_a_method_factory_sees_the_recorded_value(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")
    add = record_method(
        store, exclude_attrs=["currency"], factory=targets.new_wallet, env=(CURRENCY,)
    )(targets.Wallet.add.__wrapped__)
    assert add(targets.new_wallet(), 5.0) == "5.0 EUR"
    monkeypatch.setenv(CURRENCY, "USD")

    case = only_case(replay_store(store))

    assert case.status is ReplayStatus.PASSED, case.describe()


def test_the_case_carries_the_environment_into_its_dict(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")
    record(store, targets.in_currency, 5, env=(CURRENCY,))

    data = only_case(replay_store(store)).to_dict()

    assert data["env"] == {CURRENCY: "EUR"}
    assert data["secret_env_changed"] == []


# ------------------------------------------------------------------ secrets


def test_a_secret_is_fingerprinted_not_stored(store, monkeypatch):
    monkeypatch.setenv(API_KEY, "sk-live-0123456789abcdef")

    record(store, targets.total, [1, 2], secret_env=(API_KEY,))

    [path] = store.iter_record_paths()
    text = path.read_text()
    assert "sk-live-0123456789abcdef" not in text
    secret = json.loads(text)["secret_env"][API_KEY]
    assert secret.startswith("pbkdf2_sha256$600000$")
    assert "env" not in json.loads(text)


def test_a_secret_does_not_split_cases(store, monkeypatch):
    for key in ("key-one", "key-two"):
        monkeypatch.setenv(API_KEY, key)
        record(store, targets.total, [1, 2], secret_env=(API_KEY,))

    assert len(record_dicts(store)) == 1


@pytest.mark.parametrize(
    ("now", "changed"),
    [("sk-a", ()), ("sk-b", (API_KEY,)), (None, (API_KEY,))],
    ids=["same", "rotated", "unset"],
)
def test_replay_reports_a_changed_secret(store, monkeypatch, now, changed):
    monkeypatch.setenv(API_KEY, "sk-a")
    record(store, targets.total, [1, 2], secret_env=(API_KEY,))
    if now is None:
        monkeypatch.delenv(API_KEY)
    else:
        monkeypatch.setenv(API_KEY, now)

    case = only_case(replay_store(store))

    assert case.status is ReplayStatus.PASSED, "reported, not failed"
    assert case.secret_env_changed == changed
    assert ("secret changed" in case.describe()) == bool(changed)
    assert os.environ.get(API_KEY) == now, "never set by replay"


def test_a_secret_recorded_unset_that_is_now_set_has_changed(monkeypatch):
    monkeypatch.setenv(API_KEY, "sk-a")
    assert changed_secrets({API_KEY: None}) == (API_KEY,)
    monkeypatch.delenv(API_KEY)
    assert changed_secrets({API_KEY: None}) == ()


def test_a_fingerprint_is_salted_by_name_and_keeps_its_rounds(monkeypatch):
    assert fingerprint("A", "same") != fingerprint("B", "same")
    old = fingerprint(API_KEY, "sk-a", rounds=1_000)
    monkeypatch.setenv(API_KEY, "sk-a")
    assert changed_secrets({API_KEY: old}) == ()
    assert changed_secrets({API_KEY: "unknown$scheme"}) == (API_KEY,)


# ------------------------------------------------------------------ the names


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"env": CURRENCY}, TypeError, "sequence of variable names"),
        ({"secret_env": API_KEY}, TypeError, "sequence of variable names"),
        ({"env": [1]}, TypeError, "must be strings"),
        ({"env": [""]}, ValueError, "not an environment variable name"),
        ({"env": ["A=B"]}, ValueError, "not an environment variable name"),
        ({"env": ["REGRESSION_RECORDS"]}, ValueError, "recording switch"),
        ({"env": ["A"], "secret_env": ["A"]}, ValueError, "both env and secret_env"),
    ],
)
@pytest.mark.parametrize("decorator", [record_func, record_method])
def test_bad_names_are_refused_when_decorating(store, decorator, kwargs, error, match):
    with pytest.raises(error, match=match):
        decorator(store, **kwargs)


def test_names_given_as_a_generator_are_all_recorded(store, monkeypatch):
    monkeypatch.setenv(CURRENCY, "EUR")

    record(store, targets.in_currency, 5, env=(n for n in [CURRENCY]))

    [data] = record_dicts(store)
    assert data["env"] == {CURRENCY: "EUR"}
