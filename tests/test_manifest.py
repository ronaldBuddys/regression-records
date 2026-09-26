"""Run manifests: what a recording session lists, and under what environment."""

import json
import logging
import subprocess
import threading

import numpy as np
import pytest

from regression_records import (
    RecordStore,
    environment,
    record_func,
    record_method,
    replay,
    session,
)
from regression_records._engine import forget_recorded_calls
from regression_records.manifest import (
    iter_manifest_paths,
    load_manifest,
    package_state,
    repo_state,
)
from tests import _replay_targets as targets

# Recorded under total's real identity, so the records replay.
add_one = targets.total.__wrapped__


@pytest.fixture()
def store(tmp_path):
    return RecordStore(tmp_path / "store")


def _statuses(manifest):
    return sorted((r["name"].split(":")[1], r["status"]) for r in manifest["records"])


def test_session_lists_new_records_and_writes_a_manifest(store):
    recorded = record_func(store)(add_one)
    with session(store, context={"seed": 0}) as run:
        recorded([1])
        recorded([2])

    assert run.manifest_path == iter_manifest_paths(store)[0]
    manifest = load_manifest(run.manifest_path)
    assert manifest["run_id"] == run.run_id
    assert manifest["completed"] is True
    assert manifest["context"] == {"seed": 0}
    assert manifest["counts"] == {"new": 2, "matched": 0}
    assert _statuses(manifest) == [("total", "new"), ("total", "new")]
    listed = {r["record_path"] for r in manifest["records"]}
    on_disk = {p.relative_to(store.root).as_posix() for p in store.iter_record_paths()}
    assert listed == on_disk


def test_rerun_with_identical_behavior_lists_every_record_as_matched(store):
    recorded = record_func(store)(add_one)
    with session(store) as first:
        recorded([1])
    forget_recorded_calls()  # as a new process would
    with session(store) as second:
        recorded([1])

    assert [r.status for r in first.records] == ["new"]
    assert [r.status for r in second.records] == ["matched"]
    assert first.records[0].record_id == second.records[0].record_id
    assert len(store.iter_record_paths()) == 1
    assert len(iter_manifest_paths(store)) == 2


def test_a_repeat_within_the_process_is_matched_and_does_not_demote_new(store):
    recorded = record_func(store)(add_one)
    with session(store) as run:
        recorded([1])
        recorded([1])
    assert [r.status for r in run.records] == ["new"]

    with session(store) as again:
        recorded([1])  # caught by the per-process pre-check, before any file is read
    assert [r.status for r in again.records] == ["matched"]


def test_method_records_are_listed(store):
    bump = record_method(store)(targets.Counter.bump.__wrapped__)
    with session(store) as run:
        bump(targets.Counter(), 2)
    assert [r.name for r in run.records] == [f"{targets.__name__}:Counter.bump"]


def test_records_sent_to_another_store_are_not_listed(tmp_path, store):
    elsewhere = record_func(tmp_path / "other")(add_one)
    with session(store) as run:
        elsewhere([1])
    assert run.records == ()


def test_a_store_given_by_path_matches_the_same_store(store):
    recorded = record_func(store)(add_one)
    with session(str(store.root)) as run:
        recorded([1])
    assert len(run.records) == 1


def test_calls_from_worker_threads_are_listed(store):
    recorded = record_func(store)(add_one)
    with session(store) as run:
        worker = threading.Thread(target=recorded, args=([5],))
        worker.start()
        worker.join()
    assert len(run.records) == 1


def test_a_failing_block_still_writes_its_manifest(store):
    recorded = record_func(store)(add_one)
    with pytest.raises(RuntimeError, match="boom"), session(store) as run:
        recorded([1])
        raise RuntimeError("boom")

    manifest = load_manifest(run.manifest_path)
    assert manifest["completed"] is False
    assert manifest["error"] == "RuntimeError: boom"
    assert manifest["counts"]["new"] == 1


def test_context_that_is_not_json_is_refused_on_entry(store):
    with pytest.raises(TypeError), session(store, context={"when": object()}):
        pytest.fail("the block must not run")
    with pytest.raises(ValueError), session(store, context={"x": float("nan")}):
        pytest.fail("the block must not run")
    assert iter_manifest_paths(store) == []


def test_an_empty_session_with_recording_off_warns(store, monkeypatch, caplog):
    monkeypatch.setenv("REGRESSION_RECORDS", "0")
    recorded = record_func(store)(add_one)
    with caplog.at_level(logging.WARNING), session(store) as run:
        recorded([1])
    assert run.records == ()
    assert "recording is off" in caplog.text


def test_manifests_are_invisible_to_the_store_and_to_replay(store):
    recorded = record_func(store)(add_one)
    with session(store):
        recorded(np.arange(3))
    assert all("_runs" not in p.parts for p in store.iter_record_paths())
    assert store.prune() == []
    report = replay(store)
    assert report.passed and len(report.cases) == 1


def test_environment_describes_packages_and_flags_unknowns(caplog):
    with caplog.at_level(logging.WARNING):
        env = environment(packages=["numpy", "no-such-distribution-xyz"])
    assert env.packages["numpy"].version == np.__version__
    assert env.packages["no-such-distribution-xyz"].version is None
    assert "no-such-distribution-xyz" in caplog.text
    json.dumps(env.to_dict())


def test_package_state_reads_a_vcs_install(monkeypatch):
    class FakeDist:
        version = "0.3.0"

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": "https://github.com/o/r.git",
                    "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                }
            )

    monkeypatch.setattr(
        "regression_records.manifest.metadata.distribution", lambda name: FakeDist()
    )
    state = package_state("r")
    assert (state.version, state.url, state.vcs_commit, state.editable) == (
        "0.3.0",
        "https://github.com/o/r.git",
        "abc123",
        False,
    )


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", "code.py")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-q",
        "-m",
        "init",
    )
    return repo


def test_repo_state_flags_tracked_changes_but_not_untracked_files(repo):
    clean = repo_state(repo)
    assert len(clean.commit) == 40 and clean.dirty is False

    (repo / "scratch.md").write_text("notes\n")
    assert repo_state(repo).dirty is False

    (repo / "code.py").write_text("x = 2\n")
    dirty = repo_state(repo)
    assert dirty.dirty is True and dirty.commit == clean.commit


def test_repo_state_outside_a_repository_is_unknown(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        state = repo_state(tmp_path)
    assert (state.commit, state.dirty) == (None, None)
    assert "recorded as unknown" in caplog.text


def test_session_manifest_carries_the_environment(store, repo):
    with session(store, repos={"code": repo}, packages=["numpy"]) as run:
        pass
    env = load_manifest(run.manifest_path)["environment"]
    assert env["repos"]["code"]["dirty"] is False
    assert env["packages"]["numpy"]["version"] == np.__version__


def test_replay_report_save_round_trips(store, tmp_path, repo):
    recorded = record_func(store)(add_one)
    recorded([1])
    report = replay(store)
    path = report.save(
        tmp_path / "reports" / "replay.json",
        environment=environment(repos={"code": repo}),
        context={"label": "baseline"},
    )
    saved = json.loads(path.read_text())
    assert saved["counts"] == report.counts()
    assert saved["cases"] == report.to_dict()["cases"]
    assert saved["context"] == {"label": "baseline"}
    assert saved["environment"]["repos"]["code"]["dirty"] is False


def test_a_repeat_through_another_spelling_of_the_root_is_listed(store, monkeypatch):
    monkeypatch.chdir(store.root.parent)
    absolute = record_func(store)(add_one)
    relative = record_func(RecordStore(store.root.name))(add_one)
    with session(store) as run:
        absolute([1])
        relative([1])  # the same call key, remembered with the absolute path
    assert [r.status for r in run.records] == ["new"]
