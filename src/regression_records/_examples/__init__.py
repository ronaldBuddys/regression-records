"""Support code for the documentation and the example gallery only.

Every submodule here backs one page of the gallery under ``examples/`` or one
illustrative snippet in the hand-written docs: a function or a class that needs a real,
importable module to live in, since replay refuses anything defined in ``__main__``.

Each submodule records to ``example_store(<its own name>)``: a directory under a fresh
temporary one by default, or under ``$REGRESSION_RECORDS_EXAMPLE_STORE`` when that is
set, for a reader who wants to keep and inspect what a page recorded. Point it at an
empty directory: a page counts what it records, so a second run into the same directory
finds the first run's records and fails its checks. Nothing here ever deletes a store.

A page that starts a second Python process passes it this process's root in that
variable, so that both record into the same store rather than one temporary directory
each.

**Nothing here is public API.** Signatures, names and behaviour may change, move or
disappear between releases without notice, and without a ``CHANGELOG`` entry. Do not
import from ``regression_records._examples`` in your own code; it exists so the
examples can import real functions and classes instead of writing a module to a
temporary file at each page's top.
"""

from __future__ import annotations

import os
import tempfile
from functools import cache
from pathlib import Path

STORE_ROOT_ENV = "REGRESSION_RECORDS_EXAMPLE_STORE"
"""Environment variable naming where the examples keep their records."""


@cache
def _session_root() -> Path:
    """Return one fresh temporary directory per process, made on first use."""
    return Path(tempfile.mkdtemp(prefix="regression-records-examples-"))


def example_store(name: str) -> Path:
    """Return the store path a submodule of ``_examples`` records to.

    Parameters
    ----------
    name : str
        The submodule's own name, which becomes a directory under the root, so pages
        sharing a root still get a store each.

    Returns
    -------
    pathlib.Path
        ``$REGRESSION_RECORDS_EXAMPLE_STORE/<name>`` when that variable is set, so a
        reader can keep the records of a page they ran. Otherwise a directory under a
        fresh temporary one, made once per process: a page never replays records an
        earlier run left behind, and nothing needs deleting afterwards.
    """
    root = os.environ.get(STORE_ROOT_ENV)
    return (Path(root).expanduser() if root else _session_root()) / name
