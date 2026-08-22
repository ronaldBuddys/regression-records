"""Content-addressed filename generation for externally stored data."""

from __future__ import annotations

import hashlib


def hash_bytes(data: bytes) -> str:
    """Return a short, stable hex digest of ``data`` suitable for use in filenames."""
    return hashlib.sha256(data).hexdigest()[:16]


def content_filename(prefix: str, data: bytes, extension: str) -> str:
    """Build a content-addressed filename: ``<prefix>-<hash>.<extension>``.

    Identical content always produces the same filename, so callers can safely skip
    writing a file that already exists on disk (deduplication).
    """
    return f"{prefix}-{hash_bytes(data)}.{extension}"
