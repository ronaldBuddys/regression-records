"""Write files so that a reader never sees one half-written."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write ``path`` via ``write_fn(temporary_path)``, then move it into place.

    A store is read by name: a record's filename says which inputs it answers for, and
    a content-addressed side file's name is taken as proof of what it holds. A write
    that is interrupted part-way must therefore never leave a file under its final
    name. ``os.replace`` is atomic on POSIX and Windows when source and destination are
    on one filesystem, which a sibling temporary file guarantees.

    The temporary name keeps ``path``'s suffix at the end because ``numpy.save``
    appends ``.npy`` to any name lacking it, and starts with a dot so the store's
    ``rec-*`` globs never pick it up.
    """
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}-{path.name}")
    try:
        write_fn(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
