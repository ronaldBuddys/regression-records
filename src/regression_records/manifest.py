"""Run manifests: which records a recording session produced, and under what.

A record says what a call did, but not what code, packages or data it was made with -
and since a record's id is derived from its content, recording identical behavior
again writes nothing, so a run that confirmed an earlier recording leaves no trace in
the records at all. A :func:`session` fills that gap without touching either: every
record it sees written (``"new"``) or already on disk (``"matched"``) is listed in a
manifest, beside the environment the run had and whatever context the caller supplies
(a data version, a seed), written on exit to ``<root>/_runs/run-<run_id>.json``.

Records themselves are unchanged, so manifests can be added to, or deleted from, a store
without affecting replay. Replay walks only ``records/`` directories and never reads
them.
"""

from __future__ import annotations

import json
import logging
import platform
import secrets
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from ._atomic import atomic_write
from ._switch import recording_enabled
from ._version import __version__
from .store import RecordStore, coerce_store

logger = logging.getLogger(__name__)

#: The directory, directly under a store's root, that manifests are written to.
RUNS_DIR = "_runs"

#: Whether a record a session saw was written by it, or found already on disk.
RecordStatus = Literal["new", "matched"]


@dataclass(frozen=True)
class RepoState:
    """The git state of one repository.

    Attributes
    ----------
    path : str
        The directory asked about.
    commit : str or None
        ``HEAD``'s full SHA; None if git could not answer (not a repository, git
        missing).
    dirty : bool or None
        Whether tracked files differ from ``HEAD``, staged or not. Untracked files are
        ignored: scratch notes beside the code should not mark every run dirty. None
        when ``commit`` is.
    """

    path: str
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True)
class PackageState:
    """How one installed distribution was installed.

    Attributes
    ----------
    version : str or None
        Its version; None if it is not installed.
    url : str or None
        Where it was installed from, from its ``direct_url.json`` (PEP 610): a VCS URL
        or a local path. None for a package installed from an index.
    vcs_commit : str or None
        The commit a VCS install resolved to. For a git dependency this, not
        ``version``, says which code ran: the version is rarely bumped between commits.
    editable : bool
        Whether it is an editable install, whose code is whatever is on disk now.
    """

    version: str | None
    url: str | None = None
    vcs_commit: str | None = None
    editable: bool = False


@dataclass(frozen=True)
class Environment:
    """What a run was made with: interpreter, packages and repositories.

    Attributes
    ----------
    python : str
        ``sys.version``'s first word, e.g. ``"3.12.4"``.
    platform : str
        ``platform.platform()``.
    regression_records : str
        This library's version.
    packages : Mapping[str, PackageState]
        One entry per distribution asked about, keyed by the name given.
    repos : Mapping[str, RepoState]
        One entry per repository asked about, keyed by the name given.
    """

    python: str
    platform: str
    regression_records: str
    packages: Mapping[str, PackageState] = field(default_factory=dict)
    repos: Mapping[str, RepoState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dict.

        Returns
        -------
        dict
            Every attribute, with packages and repos as dicts of dicts.
        """
        return asdict(self)


def environment(
    *,
    repos: Mapping[str, str | Path] | None = None,
    packages: Sequence[str] = (),
) -> Environment:
    """Describe the current environment.

    Nothing here raises for a missing package, repository or git executable: the
    affected entry is recorded as unknown and a warning logged, since a manifest with
    a gap is worth more than a run that failed for want of one.

    Parameters
    ----------
    repos : Mapping[str, str or Path], optional
        Repositories whose ``HEAD`` and dirty flag to record, by a name of your
        choosing - typically the code under test.
    packages : Sequence[str], default ()
        Distribution names whose version and install source to record.

    Returns
    -------
    Environment
        The interpreter, platform, this library's version, and one entry per package
        and repository asked about.
    """
    return Environment(
        python=sys.version.split()[0],
        platform=platform.platform(),
        regression_records=__version__,
        packages={name: package_state(name) for name in packages},
        repos={name: repo_state(path) for name, path in (repos or {}).items()},
    )


def package_state(name: str) -> PackageState:
    """Return how the distribution ``name`` is installed; see :class:`PackageState`."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        logger.warning("package %r is not installed; recorded as unknown", name)
        return PackageState(version=None)
    # direct_url.json exists only for a package installed from a URL, a VCS or a
    # local path (PEP 610); from an index, the version is the whole story.
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return PackageState(version=dist.version)
    direct = json.loads(raw)
    return PackageState(
        version=dist.version,
        url=direct.get("url"),
        vcs_commit=direct.get("vcs_info", {}).get("commit_id"),
        editable=bool(direct.get("dir_info", {}).get("editable", False)),
    )


def repo_state(path: str | Path) -> RepoState:
    """Return the git ``HEAD`` and dirty flag of ``path``; see :class:`RepoState`."""
    commit = _git(path, "rev-parse", "HEAD")
    if commit is None:
        return RepoState(path=str(path), commit=None, dirty=None)
    status = _git(path, "status", "--porcelain", "--untracked-files=no")
    return RepoState(
        path=str(path),
        commit=commit,
        dirty=None if status is None else bool(status),
    )


def _git(path: str | Path, *args: str) -> str | None:
    """Run ``git -C path args``, returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning("git %s failed in %s; recorded as unknown: %s", args[0], path, e)
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class SessionRecord:
    """One record a session saw.

    Attributes
    ----------
    record_path : str
        The record file, relative to the store root, with ``/`` separators.
    record_id : str
        The record's id.
    name : str
        ``"module:qualname"`` of the recorded function.
    status : {"new", "matched"}
        ``"new"`` if this session wrote it, ``"matched"`` if the same behavior was
        already recorded - by an earlier run, a parallel worker, or earlier in this one.
    """

    record_path: str
    record_id: str
    name: str
    status: RecordStatus


@dataclass
class RecordingSession:
    """A recording session in progress; see :func:`session`.

    Attributes
    ----------
    store : RecordStore
        The store whose records are listed.
    run_id : str
        ``<UTC timestamp, to the microsecond>Z-<random suffix>``: unique per session,
        and sorting run ids sorts sessions by start time.
    context : Mapping[str, Any]
        The caller's JSON-safe context.
    environment : Environment
        Captured when the session started.
    started_at : str
        ISO 8601 UTC.
    manifest_path : Path or None
        Where the manifest was written; None until the session ends.
    """

    store: RecordStore
    run_id: str
    context: Mapping[str, Any]
    environment: Environment
    started_at: str
    manifest_path: Path | None = None
    _records: dict[str, SessionRecord] = field(default_factory=dict, repr=False)

    @property
    def records(self) -> tuple[SessionRecord, ...]:
        """Return the records seen so far, sorted by record path."""
        with _sessions_lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def _note(self, entry: SessionRecord) -> None:
        # A record this session wrote stays "new" when a later call matches it.
        with _sessions_lock:
            seen = self._records.get(entry.record_path)
            if seen is None or seen.status == "matched":
                self._records[entry.record_path] = entry

    def to_dict(self, *, finished_at: str, error: str | None) -> dict[str, Any]:
        """Convert to the manifest's JSON-safe dict.

        Parameters
        ----------
        finished_at : str
            ISO 8601 UTC.
        error : str or None
            The exception that ended the session, if one did.

        Returns
        -------
        dict
            The run id, times, whether it completed, the error, the environment, the
            context, counts of new and matched records, and every record.
        """
        records = self.records
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "completed": error is None,
            "error": error,
            "environment": self.environment.to_dict(),
            "context": dict(self.context),
            "counts": {
                status: sum(r.status == status for r in records)
                for status in ("new", "matched")
            },
            "records": [asdict(r) for r in records],
        }


# Process-wide rather than thread-local, so a decorated call made from a worker thread
# is still listed - the same choice the replay suppression makes, for the same reason.
_sessions: list[RecordingSession] = []
_sessions_lock = threading.Lock()


def note_record(
    store: RecordStore, record_path: Path, record_id: str, name: str, status: str
) -> None:
    """Tell every open session on ``store`` about one record. Called by the engine."""
    with _sessions_lock:
        active = [s for s in _sessions if _same_root(s.store, store)]
    if not active:
        return
    entry = SessionRecord(
        # Resolved on both sides: the in-process memory of recorded calls is shared by
        # every spelling of a store's root, so the path may use a different one.
        record_path=record_path.resolve().relative_to(store.root.resolve()).as_posix(),
        record_id=record_id,
        name=name,
        status=status,  # type: ignore[arg-type]
    )
    for open_session in active:
        open_session._note(entry)


def _same_root(a: RecordStore, b: RecordStore) -> bool:
    return a.root.resolve() == b.root.resolve()


@contextmanager
def session(
    store: RecordStore | str | Path,
    *,
    context: Mapping[str, Any] | None = None,
    repos: Mapping[str, str | Path] | None = None,
    packages: Sequence[str] = (),
) -> Iterator[RecordingSession]:
    """List every record made to ``store`` in the block, in a manifest written on exit.

    A session does not switch recording on: combine it with
    :func:`~regression_records.recording` or ``REGRESSION_RECORDS=1``. A warning is
    logged if it ends having seen nothing while recording is off.

    The manifest is written even when the block raises - marked ``"completed": false``
    with the error - and the exception is re-raised. With pytest-xdist each worker
    process that opens a session writes its own manifest.

    Parameters
    ----------
    store : RecordStore or str or Path
        The store to list records for. Records sent to any other store are ignored.
    context : Mapping[str, Any], optional
        Whatever else identifies the run and is not in the environment: a data
        version, a seed, a configuration name. Must be JSON-serializable.
    repos : Mapping[str, str or Path], optional
        Repositories whose commit and dirty flag to record; see :func:`environment`.
    packages : Sequence[str], default ()
        Distributions whose version and install source to record.

    Yields
    ------
    RecordingSession
        Its ``records`` fill in as the block runs; ``manifest_path`` is set on exit.

    Raises
    ------
    TypeError, ValueError
        If ``context`` is not JSON-serializable, checked on entry rather than at the
        end of a long run.

    Examples
    --------
    >>> from regression_records import recording, session  # doctest: +SKIP
    >>> with recording(), session("tests/records", context={"seed": 0}):  # doctest: +SKIP
    ...     run_pipeline()
    """
    context = dict(context or {})
    json.dumps(context, allow_nan=False)
    now = datetime.now(UTC)
    current = RecordingSession(
        store=coerce_store(store),
        run_id=f"{now.strftime('%Y%m%dT%H%M%S.%f')}Z-{secrets.token_hex(3)}",
        context=context,
        environment=environment(repos=repos, packages=packages),
        started_at=now.isoformat(),
    )
    with _sessions_lock:
        _sessions.append(current)
    error: str | None = None
    try:
        yield current
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        with _sessions_lock:
            _sessions.remove(current)
        if not current.records and not recording_enabled():
            logger.warning(
                "session %s saw no records, and recording is off: switch it on with "
                "recording() or REGRESSION_RECORDS=1",
                current.run_id,
            )
        current.manifest_path = _write_manifest(current, error)


def _write_manifest(current: RecordingSession, error: str | None) -> Path:
    runs_dir = current.store.root / RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"run-{current.run_id}.json"
    manifest = current.to_dict(finished_at=datetime.now(UTC).isoformat(), error=error)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    atomic_write(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest written by :func:`session`.

    Parameters
    ----------
    path : str or Path
        A ``run-*.json`` file, e.g. from :func:`iter_manifest_paths`.

    Returns
    -------
    dict
        The manifest as written; see :meth:`RecordingSession.to_dict`.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_manifest_paths(store: RecordStore | str | Path) -> list[Path]:
    """Return every manifest in ``store``, oldest first.

    Parameters
    ----------
    store : RecordStore or str or Path
        The store to look in.

    Returns
    -------
    list[Path]
        The ``run-*.json`` files in ``<root>/_runs``, sorted by name, which is by start
        time; empty if there are none.
    """
    runs_dir = coerce_store(store).root / RUNS_DIR
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.glob("run-*.json"))
