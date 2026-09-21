"""Content-addressed filename generation for externally stored data.

Two digest lengths are in use, and both are defined here. Short hashes name things that
are confirmed some other way or are informational: a record's filename (checked against
the full input hash inside it) and a function's code hash. Side files are named by
their content alone, so a collision would silently point a record at different data;
they get the longer hash. Side files written before that change carry a short hash of
the chunks simply concatenated; their names still resolve, and
:func:`verify_content_file` still checks them.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

from ..exceptions import CorruptRecordError

#: Hex characters of a short hash (64 bits): record filename prefixes, code hashes, and
#: side files written before :data:`CONTENT_HASH_CHARS` was introduced.
SHORT_HASH_CHARS = 16

#: Hex characters of a side file's content hash (128 bits).
CONTENT_HASH_CHARS = 32


def hash_bytes(*chunks: ReadableBuffer) -> str:
    """Return a short, stable hex digest of ``chunks`` concatenated.

    Taking several chunks rather than one joined buffer lets a caller hash a large value
    without first materializing a copy of it - this runs on every argument of every
    recorded call, so an avoidable copy of an array is an avoidable copy per call.

    Parameters
    ----------
    *chunks : bytes-like
        The data to hash, in order. Chunk boundaries are not part of the digest.

    Returns
    -------
    str
        The first :data:`SHORT_HASH_CHARS` hex characters of the sha256.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()[:SHORT_HASH_CHARS]


def content_hash(*chunks: ReadableBuffer) -> str:
    """Return the hex digest a side file is named by.

    Each chunk is prefixed with its length, so ``(b"ab", b"c")`` and ``(b"a", b"bc")``
    hash differently - an array's elements, shape and dtype cannot run into each other.

    Parameters
    ----------
    *chunks : bytes-like
        The data to hash, in order.

    Returns
    -------
    str
        The first :data:`CONTENT_HASH_CHARS` hex characters of the sha256.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(memoryview(chunk).nbytes.to_bytes(8, "little"))
        digest.update(chunk)
    return digest.hexdigest()[:CONTENT_HASH_CHARS]


def content_filename(prefix: str, *chunks: ReadableBuffer, extension: str) -> str:
    """Build a content-addressed filename: ``<prefix>-<hash>.<extension>``.

    Identical content always produces the same filename, so callers can safely skip
    writing a file that already exists on disk (deduplication).

    Parameters
    ----------
    prefix : str
        Names the kind of value, e.g. ``"arr"``.
    *chunks : bytes-like
        The content, as for :func:`content_hash`.
    extension : str
        The file extension, without the dot.

    Returns
    -------
    str
        The filename, with :func:`content_hash` of ``chunks`` as the hash.
    """
    return f"{prefix}-{content_hash(*chunks)}.{extension}"


def verify_content_file(path: Path, *chunks: ReadableBuffer) -> None:
    """Check that ``chunks`` - the content read back from ``path`` - match its name.

    Parameters
    ----------
    path : Path
        A side file named by :func:`content_filename`, or by the short hash side files
        were named with before.
    *chunks : bytes-like
        The same chunks the name was computed from, rebuilt from the file's content.

    Raises
    ------
    CorruptRecordError
        If the name holds no hash of a known length, or the hash does not match - the
        file was modified or replaced after it was written.
    """
    recorded = path.stem.rpartition("-")[2]
    if len(recorded) == CONTENT_HASH_CHARS:
        actual = content_hash(*chunks)
    elif len(recorded) == SHORT_HASH_CHARS:
        actual = hash_bytes(*chunks)
    else:
        raise CorruptRecordError(
            f"side file {path.name} is not named by a content hash, so it cannot be "
            "verified"
        )
    if actual != recorded:
        raise CorruptRecordError(
            f"side file {path} does not match the content hash in its name (it hashes "
            f"to {actual}); it was modified or replaced after it was recorded"
        )


def verified_source(path: Path, verify: bool) -> Path | io.BytesIO:
    """Return what to read a file-hashed side file from, checking it first if asked.

    For side files named by a hash of their own bytes (parquet, pickle). Verifying
    reads the file once and hands those bytes on, rather than reading it twice.

    Parameters
    ----------
    path : Path
        The side file.
    verify : bool
        Whether to check the file's bytes against its name.

    Returns
    -------
    Path or io.BytesIO
        ``path`` itself when not verifying; otherwise its verified bytes.

    Raises
    ------
    CorruptRecordError
        If verifying and the bytes do not match the name.
    """
    if not verify:
        return path
    content = path.read_bytes()
    verify_content_file(path, content)
    return io.BytesIO(content)


if TYPE_CHECKING:  # pragma: no cover
    #: Anything ``hashlib`` will accept: bytes, memoryview, or a buffer-exposing array.
    ReadableBuffer = bytes | bytearray | memoryview
