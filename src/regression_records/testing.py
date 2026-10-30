"""Run a replay as a pytest suite: one test per record, so a failure names its case."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .exceptions import CodeDriftWarning, CorruptRecordError
from .replay import _plan_replay, _replay_record
from .store import RecordStore


def replay_tests(
    store: RecordStore | str | Path | None = None, **replay_kwargs: Any
) -> Callable:
    """Build a pytest test that replays every record in ``store``, one case each.

    Bind the result to a name pytest collects, in a test module::

        # tests/test_regressions.py
        from regression_records import replay_tests

        test_regressions = replay_tests("tests/regression_data")

    The store is read when the module is collected, and each record becomes one
    parametrized case, identified as ``module:qualname@<record id>``. A case fails
    unless its replay passed, including when the record could not be loaded, imported
    or called: those are failures, never skips. A case that passed although the
    function's source has changed since recording warns with
    :class:`~regression_records.exceptions.CodeDriftWarning`.

    Parameters
    ----------
    store : RecordStore or str or Path, optional
        The store to replay; if omitted, read from ``REGRESSION_RECORDS_STORE``
        (environment or ``.env``), as for :func:`~regression_records.replay`.
    **replay_kwargs
        ``tolerance``, ``tolerance_overrides``, ``select``, ``comparator_registry`` and
        ``max_diffs``, as for :func:`~regression_records.replay.replay`.

    Returns
    -------
    Callable
        A test function taking one ``record_path`` parameter. A store with no records
        gives pytest's "empty parameter set" skip.

    Raises
    ------
    ImportError
        If pytest is not installed.
    FileNotFoundError
        If ``store`` does not exist, at collection.
    StoreNotConfiguredError
        If ``store`` is omitted and ``REGRESSION_RECORDS_STORE`` is not set.
    """
    try:
        import pytest
    except ImportError as e:  # pytest is a test-time tool, not a dependency
        raise ImportError("replay_tests needs pytest; install it to use it") from e

    record_store, options, paths = _plan_replay(store, **replay_kwargs)

    @pytest.mark.parametrize(
        "record_path", paths, ids=[_case_id(record_store, p) for p in paths]
    )
    def test_replay(record_path: Path) -> None:
        case = _replay_record(record_store, record_path, options)
        if not case.passed:
            pytest.fail(case.describe(), pytrace=False)
        if case.code_drift:
            warnings.warn(
                f"{case.name} passed, but its source has changed since it was "
                f"recorded (code_hash {case.recorded_code_hash} -> "
                f"{case.current_code_hash})",
                CodeDriftWarning,
                stacklevel=1,
            )

    return test_replay


def _case_id(store: RecordStore, path: Path) -> str:
    """Name a record for pytest: ``module:qualname@<record id>``, or its path."""
    try:
        record = store.load_record(path)
    except CorruptRecordError:
        return path.relative_to(store.root).as_posix()
    identity = record.identity
    return f"{identity.module}:{identity.qualname}@{record.record_id[:8]}"
