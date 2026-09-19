"""Content-addressed filename generation for externally stored data."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING


def hash_bytes(*chunks: ReadableBuffer) -> str:
    """Return a short, stable hex digest of ``chunks`` concatenated.

    Taking several chunks rather than one joined buffer lets a caller hash a large value
    without first materializing a copy of it - this runs on every argument of every
    recorded call, so an avoidable copy of an array is an avoidable copy per call.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()[:16]


def content_filename(prefix: str, *chunks: ReadableBuffer, extension: str) -> str:
    """Build a content-addressed filename: ``<prefix>-<hash>.<extension>``.

    Identical content always produces the same filename, so callers can safely skip
    writing a file that already exists on disk (deduplication).
    """
    return f"{prefix}-{hash_bytes(*chunks)}.{extension}"


if TYPE_CHECKING:  # pragma: no cover
    from typing import Union

    #: Anything ``hashlib`` will accept: bytes, memoryview, or a buffer-exposing array.
    ReadableBuffer = Union[bytes, bytearray, memoryview]
