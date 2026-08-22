"""Display helpers, so the examples spend their lines on the library rather than on
printing.

Every example writes to a throwaway store under the system temp directory, emptied on
each run - so an example can be re-run, or run out of order, and still show the same
output.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from regression_records import ComparisonResult

EXAMPLE_STORES = Path(tempfile.gettempdir()) / "regression-records-examples"


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


def show_record(
    root: Path,
    *,
    keys: Optional[list[str]] = None,
    select: Optional[Callable[[dict], bool]] = None,
) -> None:
    """Pretty-print one record from the store, optionally only the keys named.

    ``select`` picks which record when the store holds several; without it, the first.
    """
    for path in record_paths(root):
        record = json.loads(path.read_text())
        if select is None or select(record):
            heading(path.name)
            print(json.dumps({k: record[k] for k in keys} if keys else record, indent=2))
            return
    raise LookupError(f"no matching record under {root}")


def show_result(result: Optional[ComparisonResult]) -> None:
    """Print a comparison result: the summary, then each diff's kind, path and values."""
    if result is None:
        print("no comparison ran (the call was recorded, not validated)")
        return
    print(result.summary())
    for diff in result.diffs:
        print(f"  {diff.kind:<16} {diff.path}")
        print(f"      expected {diff.expected} | actual {diff.actual}")
        if diff.tolerance_used is not None:
            tol = diff.tolerance_used
            print(f"      rtol={tol.rtol} atol={tol.atol} max_abs_diff={tol.max_abs_diff}")
