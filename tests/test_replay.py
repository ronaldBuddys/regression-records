import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from regression_records import (
    RecordStore,
    ReplayStatus,
    ToleranceConfig,
    record_func,
    record_method,
    replay,
)

from . import _replay_targets as targets

MODULE = targets.__name__
V1_STORE = Path(__file__).parent / "fixtures" / "store_v1"


@pytest.fixture()
def store(tmp_path):
    return RecordStore(tmp_path / "records")


def record(store, target, *args, **kwargs):
    """Record one call of ``target`` into ``store``, under ``target``'s identity."""
    return record_func(store)(target.__wrapped__)(*args, **kwargs)


def only_case(report):
    assert len(report.cases) == 1
    return report.cases[0]


def record_files(store):
    return sorted(store.root.rglob("*.json"))


def rewrite_record(store, edit):
    """Apply ``edit`` to the store's only record file's parsed JSON."""
    (path,) = store.iter_record_paths()
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


# ------------------------------------------------------------------- the statuses


def test_an_unchanged_function_passes(store):
    record(store, targets.total, [1, 2, 3], factor=0.5)

    report = replay(store)

    case = only_case(report)
    assert case.status is ReplayStatus.PASSED
    assert report.passed
    assert case.name == f"{MODULE}:total"
    assert case.comparison.passed
    assert not case.code_drift


def test_a_changed_result_differs(store, monkeypatch):
    record(store, targets.total, [1, 2, 3])
    monkeypatch.setattr(targets, "scale", lambda v: np.asarray(v, dtype=float) * 3)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.DIFFERS
    assert [d.path for d in case.comparison.diffs] == ["$.output.total"]


def test_raising_where_the_record_returned_is_reported_as_raised(store, monkeypatch):
    record(store, targets.total, [1, 2, 3])

    def broken(values):
        raise RuntimeError("no database here")

    monkeypatch.setattr(targets, "scale", broken)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.RAISED
    assert "RuntimeError: no database here" in case.error


def test_a_recorded_exception_raised_again_passes(store):
    with pytest.raises(ValueError):
        record(store, targets.parse, "x")

    assert only_case(replay(store)).status is ReplayStatus.PASSED


def raise_type_error(text):
    raise TypeError("other")


@pytest.mark.parametrize(
    "now", [raise_type_error, lambda text: 0], ids=["raises-another-type", "returns"]
)
def test_a_recorded_exception_that_changed_differs(store, monkeypatch, now):
    with pytest.raises(ValueError):
        record(store, targets.parse, "x")
    monkeypatch.setattr(targets, "to_int", now)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.DIFFERS
    assert case.comparison.diffs[0].kind == "exception_mismatch"


def test_a_function_that_no_longer_exists_is_not_importable(store, monkeypatch):
    record(store, targets.total, [1])
    monkeypatch.delattr(targets, "total")

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_IMPORTABLE
    assert "no attribute 'total'" in case.error


def test_a_module_that_no_longer_exists_is_not_importable(store):
    record(store, targets.total, [1])
    rewrite_record(store, lambda d: d["identity"].update(module="no_such_module"))

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_IMPORTABLE
    assert "no_such_module" in case.error


def test_a_name_that_is_no_longer_recorded_is_not_called(store):
    record_func(store)(targets.undecorated)(1)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_REPLAYABLE
    assert "not decorated" in case.error


def test_a_name_holding_another_functions_recorder_is_not_called(store):
    record(store, targets.total, [1])
    rewrite_record(store, lambda d: d["identity"].update(qualname="alias"))

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_REPLAYABLE
    assert "recorder of" in case.error


def record_call_of(store, method, receiver, *args, **options):
    """Record ``receiver.method(*args)`` into ``store``, under ``method``'s identity."""
    info = method.__regression_records__
    options.setdefault("factory", info.factory)
    if info.state_spec.exclude:
        options.setdefault("exclude_attrs", info.state_spec.exclude)
    return record_method(store, **options)(method.__wrapped__)(receiver, *args)


def test_a_method_replays_from_its_recorded_state(store):
    counter = targets.Counter()
    counter.count = 5
    record_call_of(store, targets.Counter.bump, counter, 2)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.PASSED, case.describe()
    assert case.name == f"{MODULE}:Counter.bump"


@pytest.mark.parametrize(
    ("method", "receiver", "args"),
    [
        (targets.SlottedCounter.bump, targets.SlottedCounter(), (2,)),
        (targets.FrozenPoint.norm, targets.FrozenPoint(3.0, 4.0), ()),
    ],
    ids=["slots", "frozen-slots-dataclass"],
)
def test_a_slotted_or_frozen_receiver_replays(store, method, receiver, args):
    record_call_of(store, method, receiver, *args)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.PASSED, case.describe()


def test_a_named_tuple_whose_fields_changed_is_not_loadable(store):
    from ._models import Quote

    record(store, targets.spread_of, Quote(99.5, 100.5))

    def rename_ask(data):
        fields = data["invocation"]["args"][0]["attrs"]
        fields["last"] = fields.pop("ask")

    rewrite_record(store, rename_ask)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_LOADABLE
    assert "fields of 'tests._models.Quote' changed" in case.error


def test_a_changed_side_effect_differs_even_when_the_result_matches(store, monkeypatch):
    """The regression a method record exists to catch: same answer, wrong mutation."""
    record_call_of(store, targets.Widget.resize, targets.Widget(10), 20)
    monkeypatch.setattr(targets, "KEEP_HISTORY", False)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.DIFFERS
    assert [d.path for d in case.comparison.diffs] == ["$.state_after.history"]


def test_a_partial_state_is_replayed_on_a_receiver_from_the_factory(store):
    account = targets.new_account()
    account.balance = 10.0
    record_call_of(store, targets.Account.deposit, account, 5.0)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.PASSED, case.describe()


def test_a_factory_receiver_keeps_attributes_the_spec_now_excludes(store):
    """An old record holding an attribute now excluded must not overwrite it."""
    account = targets.new_account()
    account.balance = 10.0
    # Recorded before ``connection`` was excluded: the record holds a stale copy of it.
    record_call_of(store, targets.Account.deposit, account, 5.0, exclude_attrs=())

    case = only_case(replay(store))

    # The factory's live connection is kept; the recorded balance is restored. The
    # recorded state_after still holds a connection the current spec leaves out.
    assert case.status is ReplayStatus.DIFFERS, case.describe()
    assert [d.path for d in case.comparison.diffs] == ["$.state_after.connection"]


def test_a_factory_building_the_wrong_class_is_not_replayable(store):
    record_call_of(
        store,
        targets.Misbuilt.deposit,
        targets.Misbuilt(targets.Connection()),
        1.0,
        factory=targets.new_account,
    )

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_REPLAYABLE
    assert "not an instance of Misbuilt" in case.error


def test_a_class_with_its_own_state_protocol_is_rebuilt_through_it(store):
    record_call_of(store, targets.Stateful.double, targets.Stateful(3))

    case = only_case(replay(store))

    assert case.status is ReplayStatus.PASSED, case.describe()


def test_a_method_recorded_as_a_function_is_not_replayed(store):
    record_call_of(store, targets.Counter.bump, targets.Counter(), 1)
    rewrite_record(store, lambda d: d["identity"].update(kind="function"))

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_REPLAYABLE
    assert "recorded as a function" in case.error


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda path: path.write_text("{not json"),
        # Neither an output nor an exception: the record cannot say what the call did.
        lambda path: path.write_text(
            json.dumps(
                {k: v for k, v in json.loads(path.read_text()).items() if k != "output"}
            )
        ),
    ],
    ids=["invalid-json", "no-output-or-exception"],
)
def test_an_unreadable_record_is_not_loadable(store, corrupt):
    record(store, targets.total, [1])
    (path,) = store.iter_record_paths()
    corrupt(path)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_LOADABLE
    assert case.identity is None
    assert case.name == case.record_path


def test_a_missing_side_file_is_not_loadable(store):
    record(store, targets.weights, np.arange(5.0))
    for data_file in store.root.rglob("data/*"):
        data_file.unlink()

    case = only_case(replay(store))

    assert case.status is ReplayStatus.NOT_LOADABLE
    assert "arguments" in case.error
    assert case.identity is not None


def test_array_arguments_and_results_replay(store):
    record(store, targets.weights, np.arange(5.0))

    assert only_case(replay(store)).status is ReplayStatus.PASSED


# ----------------------------------------------------------------------- tolerance


@pytest.fixture()
def slightly_off(store, monkeypatch):
    record(store, targets.total, [1, 2, 3])
    monkeypatch.setattr(
        targets, "scale", lambda v: np.asarray(v, dtype=float) * 2 * (1 + 1e-6)
    )


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({}, ReplayStatus.DIFFERS),
        ({"tolerance": ToleranceConfig(rtol=1e-5)}, ReplayStatus.PASSED),
        (
            {"tolerance_overrides": {f"{MODULE}:tot*": ToleranceConfig(rtol=1e-5)}},
            ReplayStatus.PASSED,
        ),
        (
            {"tolerance_overrides": {"elsewhere:*": ToleranceConfig(rtol=1e-5)}},
            ReplayStatus.DIFFERS,
        ),
    ],
    ids=["default", "tolerance", "matching-override", "other-override"],
)
def test_tolerances(store, slightly_off, kwargs, status):
    assert only_case(replay(store, **kwargs)).status is status


# -------------------------------------------------------------------- selecting


@pytest.mark.parametrize(
    "select",
    ["*:parse", lambda record: record.identity.qualname == "parse"],
    ids=["glob", "predicate"],
)
def test_select_keeps_matching_records(store, select):
    record(store, targets.total, [1])
    with pytest.raises(ValueError):
        record(store, targets.parse, "x")

    case = only_case(replay(store, select=select))

    assert case.identity.qualname == "parse"


def test_an_unreadable_record_is_selected_by_its_location(store):
    record(store, targets.total, [1])
    (path,) = store.iter_record_paths()
    path.write_text("{not json")

    assert only_case(replay(store, select=f"{MODULE}:total")).status is (
        ReplayStatus.NOT_LOADABLE
    )
    assert replay(store, select="*:parse").cases == ()
    # A predicate cannot look at a record it cannot read, so the record is kept.
    assert len(replay(store, select=lambda record: False).cases) == 1


# --------------------------------------------------------------- replay only reads


@pytest.mark.parametrize("target", [targets.nested, targets.threaded])
def test_replay_never_writes(store, tmp_path, monkeypatch, target):
    monkeypatch.setattr(targets.INNER_STORE, "root", tmp_path / "inner")
    monkeypatch.setenv("REGRESSION_RECORDS", "1")
    record(store, target, 1)
    # Recording ran the function for real, so the inner call recorded too.
    assert len(record_files(targets.INNER_STORE)) == 1
    # An empty store for the inner call, where anything it did write would be new.
    monkeypatch.setattr(targets.INNER_STORE, "root", tmp_path / "inner-during-replay")
    before = record_files(store)

    case = only_case(replay(store))

    assert case.status is ReplayStatus.PASSED
    assert record_files(store) == before
    assert record_files(targets.INNER_STORE) == []


def test_the_switch_is_restored_after_replay(store, monkeypatch):
    monkeypatch.setenv("REGRESSION_RECORDS", "1")
    record(store, targets.total, [1])

    replay(store)

    assert os.environ["REGRESSION_RECORDS"] == "1"


# ------------------------------------------------------------------------- the report


def test_code_drift_is_reported_not_failed(store):
    record(store, targets.total, [1])
    rewrite_record(store, lambda d: d["identity"].update(code_hash="0" * 16))

    case = only_case(replay(store))

    assert case.status is ReplayStatus.PASSED
    assert case.code_drift
    assert case.recorded_code_hash == "0" * 16
    assert case.current_code_hash not in ("0" * 16, "unknown")
    assert "code changed" in case.describe()


def test_diffs_are_capped_per_case(store, monkeypatch):
    record(store, targets.spread, 10)
    monkeypatch.setattr(targets, "scale", lambda v: np.asarray(v, dtype=float) * 3)

    case = only_case(replay(store, max_diffs=3))

    assert len(case.comparison.diffs) == 3
    assert case.diffs_truncated == 7
    assert "7 further diff(s)" in case.describe()


def test_the_report_summarizes_and_serializes(store, monkeypatch):
    record(store, targets.total, [1])
    record(store, targets.spread, 2)
    record_func(store)(targets.undecorated)(1)
    monkeypatch.setattr(targets, "scale", lambda v: np.asarray(v, dtype=float) * 3)

    report = replay(store)

    assert not report.passed
    assert report.counts() == {
        "passed": 0,
        "differs": 2,
        "raised": 0,
        "not_importable": 0,
        "not_replayable": 1,
        "not_loadable": 0,
    }
    summary = report.summary()
    assert "3 record(s)" in summary
    assert f"{MODULE}:undecorated" in summary
    as_json = json.loads(json.dumps(report.to_dict()))
    assert as_json["counts"]["differs"] == 2
    assert {c["status"] for c in as_json["cases"]} == {"differs", "not_replayable"}


def test_a_missing_store_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay(tmp_path / "typo")


def test_an_empty_store_replays_nothing(tmp_path):
    report = replay(tmp_path)

    assert report.cases == ()
    assert report.passed
    assert "nothing to replay" in report.summary()


# ------------------------------------------------------------ records already on disk


V1_SOURCE = """
    import numpy as np

    from regression_records import record_func


    @record_func("unused", enabled=False)
    def non_finite(x):
        return {"nan": float("nan"), "inf": float("inf"), "ninf": -float("inf"), "x": x}


    @record_func("unused", enabled=False)
    def numpy_non_finite(x):
        return [np.float64("nan"), np.float64("inf"), np.float32("-inf"), x]


    @record_func("unused", enabled=False)
    def raises(x):
        raise ValueError(f"bad {x}")
"""


def test_schema_version_1_records_replay_without_migration(tmp_path, monkeypatch):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "v1_fixture_source.py").write_text(textwrap.dedent(V1_SOURCE))
    monkeypatch.syspath_prepend(str(source_dir))
    monkeypatch.delitem(sys.modules, "v1_fixture_source", raising=False)
    store_root = tmp_path / "store"
    shutil.copytree(V1_STORE, store_root)

    report = replay(store_root)

    assert [c.identity.qualname for c in report.cases] == [
        "non_finite",
        "numpy_non_finite",
        "raises",
    ]
    assert report.passed, report.summary()
