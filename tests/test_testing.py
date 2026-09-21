import builtins
import json

import numpy as np
import pytest

from regression_records import RecordStore, record_func, replay_tests

from . import _replay_targets as targets

MODULE = targets.__name__


@pytest.fixture()
def store(tmp_path):
    return RecordStore(tmp_path / "records")


def record(store, target, *args):
    return record_func(store)(target.__wrapped__)(*args)


def run_suite(pytester, store, *args, **replay_kwargs):
    """Run pytest on a module holding ``test_regressions = replay_tests(store)``."""
    pytester.makepyfile(
        test_regressions=f"""
        from regression_records import replay_tests

        test_regressions = replay_tests({str(store.root)!r}, **{replay_kwargs!r})
        """
    )
    return pytester.runpytest("-p", "no:cacheprovider", "-rA", *args)


def test_each_record_is_its_own_test(pytester, store):
    record(store, targets.total, [1, 2])
    record(store, targets.total, [3])

    result = run_suite(pytester, store)

    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines([f"PASSED*test_regressions?{MODULE}:total@*"])


def test_a_failing_case_names_its_function_and_diff(pytester, store, monkeypatch):
    record(store, targets.total, [1, 2])
    record(store, targets.parse, "7")
    monkeypatch.setattr(targets, "scale", lambda v: np.asarray(v, dtype=float) * 3)

    result = run_suite(pytester, store)

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [f"*differs: {MODULE}:total record *", "*$.output.total*"]
    )


def test_an_unloadable_record_fails_rather_than_skips(pytester, store):
    record(store, targets.total, [1])
    (path,) = store.iter_record_paths()
    path.write_text("{not json")

    result = run_suite(pytester, store)

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*not_loadable*"])


def test_code_drift_on_a_passing_case_warns(pytester, store):
    record(store, targets.total, [1])
    (path,) = store.iter_record_paths()
    data = json.loads(path.read_text())
    data["identity"]["code_hash"] = "0" * 16
    path.write_text(json.dumps(data))

    result = run_suite(pytester, store)

    result.assert_outcomes(passed=1, warnings=1)
    result.stdout.fnmatch_lines(
        [f"*CodeDriftWarning: {MODULE}:total passed, but its source has changed*"]
    )


def test_replay_arguments_are_passed_through(pytester, store):
    record(store, targets.total, [1])
    record(store, targets.parse, "7")

    result = run_suite(pytester, store, select="*:parse")

    result.assert_outcomes(passed=1)


def test_an_empty_store_is_skipped(pytester, tmp_path):
    result = run_suite(pytester, RecordStore(tmp_path))

    result.assert_outcomes(skipped=1)


def test_a_missing_store_fails_collection(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay_tests(tmp_path / "typo")


def test_pytest_missing_is_explained(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def no_pytest(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError("No module named 'pytest'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pytest)

    with pytest.raises(ImportError, match="replay_tests needs pytest"):
        replay_tests(tmp_path)
