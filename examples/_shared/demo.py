"""Display helpers, so the examples spend their lines on the library rather than on
printing.

Every example writes to a throwaway store under the system temp directory, emptied on
each run - so an example can be re-run, or run out of order, and still show the same
output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from regression_records import ComparisonResult, ReplayReport

EXAMPLE_STORES = Path(tempfile.gettempdir()) / "regression-records-examples"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent


def fresh_store(name: str) -> Path:
    """An empty store directory for this example, wiping any previous run's records."""
    root = EXAMPLE_STORES / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def show_tree(root: Path) -> None:
    """Print the store's contents, relative to its root."""
    heading(f"{root}")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(root)}")


def record_paths(root: Path) -> list[Path]:
    """Every record file in the store, in a stable order."""
    return sorted(root.rglob("rec-*.json"))


def show_counts(root: Path) -> None:
    """Print how many records each function has in the store."""
    counts: dict[str, int] = {}
    for path in record_paths(root):
        # A record sits at <module parts>/<qualname>/records/rec-*.json.
        parts = path.parent.parent.relative_to(root).parts
        name = f"{'.'.join(parts[:-1])}:{parts[-1]}"
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        print("  (no records)")
    for name, count in sorted(counts.items()):
        print(f"  {count} record(s)  {name}")


def show_record(
    root: Path,
    *,
    keys: list[str] | None = None,
    select: Callable[[dict], bool] | None = None,
) -> None:
    """Pretty-print one record from the store, optionally only the keys named.

    ``select`` picks which record when the store holds several; without it, the first.
    """
    for path in record_paths(root):
        record = json.loads(path.read_text())
        if select is None or select(record):
            heading(path.name)
            print(
                json.dumps({k: record[k] for k in keys} if keys else record, indent=2)
            )
            return
    raise LookupError(f"no matching record under {root}")


def show_result(result: ComparisonResult | None) -> None:
    """Print a comparison result: the summary, then each diff's kind, path and values."""
    if result is None:
        print("no comparison ran (the record could not be replayed)")
        return
    print(result.summary())
    for diff in result.diffs:
        print(f"  {diff.kind:<16} {diff.path}")
        print(f"      expected {diff.expected} | actual {diff.actual}")
        if diff.tolerance_used is not None:
            tol = diff.tolerance_used
            print(
                f"      rtol={tol.rtol} atol={tol.atol} max_abs_diff={tol.max_abs_diff}"
            )


def show_cases(report: ReplayReport) -> None:
    """Print one line per replayed record: its status, function, record id and drift."""
    for case in report.cases:
        record = case.record_id[:8] if case.record_id else "-"
        drift = "  (code drift)" if case.code_drift else ""
        print(f"  {case.status.value:<14} {case.name}  record {record}{drift}")


def run_in_fresh_process(code: str, **env: str) -> None:
    """Run ``code`` in a new Python process, with ``env`` added, and print its output.

    The new process can import ``_shared``, as your own package would be importable.
    It shares nothing with this one: not its module settings, and not its memory of
    which calls it has already recorded.
    """
    path = [str(EXAMPLES_DIR), os.environ.get("PYTHONPATH", "")]
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, path)), **env},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"the fresh process failed:\n{result.stderr}")
