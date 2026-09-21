"""RecordStore: the on-disk layout for regression records.

Records for one function/method identity live under:

    <root>/<module/as/dirs>/<Qualname>/records/rec-<input_hash>-<record_id>.json
    <root>/<module/as/dirs>/<Qualname>/data/<content-addressed files>

Paths referenced from a record's JSON are relative to that record's own ``records/``
directory, so the whole store (or a single identity directory) can be freely moved or
copied as a unit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from ._atomic import atomic_write
from .exceptions import CorruptRecordError, SerializationError
from .identity import FunctionIdentity
from .records import RegressionRecord
from .serialization import default_registry as _default_registry
from .serialization.base import (
    DeserializationContext,
    SerializationContext,
    SerializerRegistry,
)
from .serialization.naming import SHORT_HASH_CHARS

logger = logging.getLogger(__name__)

#: How much of the record id goes into a record's filename. The input hash leads, so
#: that looking a case up is a glob rather than a scan; it is truncated to a short hash
#: (the full one inside the file is what a lookup confirms). The record id follows to
#: keep distinct recordings of the same inputs (e.g. after an intentional behavior
#: change) from colliding.
_ID_CHARS = 8


def record_filename_prefix(input_hash: str) -> str:
    """Return the shared filename prefix of every record with this input hash."""
    return f"rec-{input_hash[:SHORT_HASH_CHARS]}-"


def record_filename(input_hash: str, record_id: str) -> str:
    """Return the on-disk filename for a record with these identifiers."""
    return f"{record_filename_prefix(input_hash)}{record_id[:_ID_CHARS]}.json"


def _write_record(path: Path, record: RegressionRecord) -> None:
    """Write ``record`` as JSON at ``path``, atomically.

    Raises ``SerializationError`` if a record of schema version 2 or later holds a NaN
    or infinite float outside the tagged encoding.
    """
    # From version 2 a record is strict JSON: the tree walk tags non-finite floats, so
    # a bare one can only come from a custom serializer's payload. A version 1 record
    # being rewritten in place (by rename_identity) may hold them and keeps its format.
    strict = record.schema.schema_version >= 2
    try:
        text = json.dumps(
            record.to_dict(), indent=2, sort_keys=True, allow_nan=not strict
        )
    except ValueError as e:
        raise SerializationError(
            f"record for {record.identity.module}:{record.identity.qualname} holds a "
            "NaN or infinite float that strict JSON cannot represent; a custom "
            "Serializer's store() must not return one (serialize nested values with "
            f"to_json_tree, or encode them as strings): {e}"
        ) from e
    # UTF-8 explicitly: the platform default is locale-dependent, and a store is meant
    # to be committed and read on other machines.
    atomic_write(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


class RecordStore:
    """Read and write regression records on disk.

    Parameters
    ----------
    root : str or Path
        The directory holding the store. It need not exist: it is created by the first
        :meth:`save`, and reading from a missing store finds no records.
    serializer_registry : SerializerRegistry, optional
        The serializers used by the contexts this store hands out;
        :data:`regression_records.serialization.default_registry` when omitted.
    verify_side_files : bool, default False
        Check every side file against the content hash in its name as it is loaded,
        raising ``CorruptRecordError`` for one that was modified or replaced. Off by
        default: it hashes every file read.

    Attributes
    ----------
    root : Path
        ``root`` as a ``Path``.
    registry : SerializerRegistry
        The registry in use.
    verify_side_files : bool
        Whether the deserialization contexts this store hands out verify side files.

    Notes
    -----
    Every method taking a :class:`~regression_records.identity.FunctionIdentity` raises
    ``ValueError`` if one of its ``path_parts`` is not a single directory name (it is
    empty, ``"."``, ``".."`` or contains a path separator), since the identity is used
    as a path beneath ``root``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        serializer_registry: SerializerRegistry | None = None,
        verify_side_files: bool = False,
    ):
        self.root = Path(root)
        self.registry = serializer_registry or _default_registry
        self.verify_side_files = verify_side_files

    def _identity_dir(self, identity: FunctionIdentity) -> Path:
        # An identity's name becomes a directory, and ``name=`` lets a caller choose
        # it. Each part must stay one directory level: a separator or ".." would walk
        # out of the store, or into the directory of a different identity.
        for part in identity.path_parts:
            if part in ("", ".", "..") or "/" in part or "\\" in part:
                raise ValueError(
                    f"identity {identity.module}:{identity.qualname} cannot be used as "
                    f"a store path: {part!r} is not a single directory name. Choose a "
                    "name without path separators or '..'."
                )
        return self.root.joinpath(*identity.path_parts)

    def _records_dir(self, identity: FunctionIdentity) -> Path:
        return self._identity_dir(identity) / "records"

    def _data_dir(self, identity: FunctionIdentity) -> Path:
        return self._identity_dir(identity) / "data"

    def serialization_context(
        self, identity: FunctionIdentity, *, deferred: bool = False
    ) -> SerializationContext:
        """Return a context that writes side files to this identity's ``data/``.

        Parameters
        ----------
        identity : FunctionIdentity
            The function or method whose values are about to be serialized.
        deferred : bool, default False
            Buffer side-file writes until the context's ``flush()`` is called, for a
            value that may turn out not to be kept.

        Returns
        -------
        SerializationContext
            Uses this store's registry, and builds references as ``"../data/<file>"`` -
            relative to the identity's ``records/`` directory.

        Raises
        ------
        ValueError
            If ``identity`` cannot be used as a store path.
        """
        # data/ is a sibling of records/, so references stored in a record's JSON must
        # be prefixed to resolve correctly relative to the records/ directory.
        return SerializationContext(
            data_dir=self._data_dir(identity),
            registry=self.registry,
            ref_prefix="../data/",
            deferred=deferred,
        )

    def deserialization_context(
        self, identity: FunctionIdentity
    ) -> DeserializationContext:
        """Return a context resolving references against the ``records/`` directory.

        Parameters
        ----------
        identity : FunctionIdentity
            The function or method whose records are about to be loaded.

        Returns
        -------
        DeserializationContext
            Uses this store's registry, refuses any reference that resolves outside
            the identity's own directory, and verifies side files when the store was
            built with ``verify_side_files=True``.

        Raises
        ------
        ValueError
            If ``identity`` cannot be used as a store path.
        """
        # External files are referenced relative to records/ (e.g. "../data/arr-....npy").
        return DeserializationContext(
            base_dir=self._records_dir(identity),
            registry=self.registry,
            root=self._identity_dir(identity),
            verify=self.verify_side_files,
        )

    def _record_path(
        self, identity: FunctionIdentity, record: RegressionRecord
    ) -> Path:
        return self._records_dir(identity) / record_filename(
            record.input_hash, record.record_id
        )

    def save(self, record: RegressionRecord, *, overwrite: bool = False) -> Path:
        """Write ``record`` to disk. Raises ``FileExistsError`` unless ``overwrite=True``.

        Parameters
        ----------
        record : RegressionRecord
            The record to write, under ``record.identity``. Its side files are not
            written here: flush the serialization context that produced them.
        overwrite : bool, default False
            Replace the file if this record's filename is already taken.

        Returns
        -------
        Path
            The JSON file written, in the identity's ``records/`` directory; see the
            module docstring for the layout.

        Raises
        ------
        FileExistsError
            If that file exists and ``overwrite`` is false. The name is derived from
            the input hash and the record id, so this means saving the same record
            twice, not recording the same inputs twice.
        SerializationError
            If the record holds a NaN or infinite float outside the tagged encoding,
            which only a custom serializer's payload can produce.
        ValueError
            If ``record.identity`` cannot be used as a store path.
        """
        records_dir = self._records_dir(record.identity)
        records_dir.mkdir(parents=True, exist_ok=True)
        path = self._record_path(record.identity, record)
        if path.exists() and not overwrite:
            raise FileExistsError(f"record file already exists: {path}")
        _write_record(path, record)
        return path

    def _load_record_file(self, path: Path) -> RegressionRecord:
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CorruptRecordError(f"could not read record file {path}: {e}") from e
        try:
            return RegressionRecord.from_dict(raw)
        except CorruptRecordError as e:
            raise CorruptRecordError(f"record file {path}: {e}") from e
        except (KeyError, TypeError) as e:
            raise CorruptRecordError(f"record file {path} is malformed: {e}") from e

    def list_records(self, identity: FunctionIdentity) -> list[RegressionRecord]:
        """Return all historical records for ``identity``, ordered by filename.

        Filenames start with the input hash, so the order is stable but not
        chronological; sort on ``created_at`` for that.

        Parameters
        ----------
        identity : FunctionIdentity
            The function or method to list. Only ``module`` and ``qualname`` locate the
            records; ``code_hash`` and ``signature`` are not compared.

        Returns
        -------
        list[RegressionRecord]
            Every record on disk for ``identity``; empty if there is none.

        Raises
        ------
        CorruptRecordError
            If any record file cannot be read, is not valid JSON, lacks a required
            field, or was written by a newer schema version.
        ValueError
            If ``identity`` cannot be used as a store path.
        """
        records_dir = self._records_dir(identity)
        if not records_dir.exists():
            return []
        return [
            self._load_record_file(path)
            for path in sorted(records_dir.glob("rec-*.json"))
        ]

    def find_by_input_hash(
        self, identity: FunctionIdentity, input_hash: str
    ) -> RegressionRecord | None:
        """Return the record for ``identity`` whose recorded inputs match exactly.

        Because the input hash is part of the filename, this is a directory glob rather
        than a scan of every historical record - it stays cheap as a function
        accumulates recorded cases, and it runs on every validated call.

        Parameters
        ----------
        identity : FunctionIdentity
            The function or method that was called.
        input_hash : str
            The full 64-character hex digest from
            :func:`~regression_records.matching.compute_input_hash`.

        Returns
        -------
        RegressionRecord or None
            The matching record, or None if these inputs were never recorded. Should
            several records share the hash, the first by filename.

        Raises
        ------
        CorruptRecordError
            If a candidate record file cannot be read or is malformed.
        ValueError
            If ``identity`` cannot be used as a store path.
        """
        records_dir = self._records_dir(identity)
        if not records_dir.exists():
            return None
        for path in sorted(
            records_dir.glob(f"{record_filename_prefix(input_hash)}*.json")
        ):
            record = self._load_record_file(path)
            # The filename holds a truncated hash; confirm against the full one inside.
            if record.input_hash == input_hash:
                return record
        return None

    def has_records(self, identity: FunctionIdentity) -> bool:
        """Whether any record exists for ``identity``, without parsing any of them.

        Parameters
        ----------
        identity : FunctionIdentity
            The function or method to look for.

        Returns
        -------
        bool
            True if at least one ``rec-*.json`` file exists, readable or not.

        Raises
        ------
        ValueError
            If ``identity`` cannot be used as a store path.
        """
        records_dir = self._records_dir(identity)
        return records_dir.exists() and any(records_dir.glob("rec-*.json"))

    def rename_identity(self, old: FunctionIdentity, new: FunctionIdentity) -> None:
        """Move all records for ``old`` to live under ``new``'s directory.

        A manual escape hatch for the case identity matching cannot resolve
        automatically: a function or method moved to a different module, or was renamed.

        Each moved record's stored ``module``/``qualname`` are updated to match, so the
        directory and the files inside it don't disagree. The rest of the recorded
        identity - ``kind``, ``code_hash``, ``signature`` - describes the code each
        record was *captured* from and is left alone, so moving a function doesn't erase
        the drift warning you would get if its implementation had also changed.

        Side files are not touched: records reference them relatively, so they move with
        the directory and keep resolving.

        Parameters
        ----------
        old : FunctionIdentity
            The identity the records are currently stored under.
        new : FunctionIdentity
            The identity to store them under. Only its ``module`` and ``qualname`` are
            used.

        Raises
        ------
        FileNotFoundError
            If nothing is stored for ``old``.
        FileExistsError
            If anything is already stored for ``new``; the two are never merged.
        CorruptRecordError
            If a moved record cannot be read back to be updated. The directory has
            been moved by then, and the records before that one rewritten.
        ValueError
            If either identity cannot be used as a store path.
        """
        old_dir = self._identity_dir(old)
        new_dir = self._identity_dir(new)
        if not old_dir.exists():
            raise FileNotFoundError(f"no records exist for identity at {old_dir}")
        if new_dir.exists():
            raise FileExistsError(
                f"records already exist for the new identity at {new_dir}"
            )
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)

        for path in sorted((new_dir / "records").glob("rec-*.json")):
            record = self._load_record_file(path)
            record.identity = replace(
                record.identity, module=new.module, qualname=new.qualname
            )
            _write_record(path, record)

    def prune(
        self, identity: FunctionIdentity | None = None, *, dry_run: bool = True
    ) -> list[Path]:
        """List, and with ``dry_run=False`` delete, data files no record references.

        A data file is kept while its name appears in any record file in the sibling
        ``records/`` directory - read as text, not parsed, so a record that no longer
        loads still protects its files, and so does any serializer's payload layout.
        Records themselves are never touched: deleting one is a manual act, and this
        collects what it leaves behind (as well as files from an overwritten record or
        an interrupted save).

        Do not prune a store while something is recording to it: a side file is
        written before the record that references it.

        Parameters
        ----------
        identity : FunctionIdentity, optional
            Prune only this function's ``data/``. When omitted, every identity in the
            store.
        dry_run : bool, default True
            Only report. Pass False to delete.

        Returns
        -------
        list[Path]
            The unreferenced files - deleted, unless ``dry_run`` - sorted by path.
            Empty when there is nothing to prune or no store.

        Raises
        ------
        ValueError
            If ``identity`` cannot be used as a store path.
        """
        if identity is not None:
            identity_dirs = [self._identity_dir(identity)]
        else:
            # An identity's data/ holds files; a module directory that happens to be
            # named "data" holds only directories.
            identity_dirs = sorted(
                {
                    path.parent
                    for path in self.root.rglob("data")
                    if path.is_dir() and any(p.is_file() for p in path.iterdir())
                }
            )

        unreferenced = [
            path for directory in identity_dirs for path in _unreferenced(directory)
        ]
        for path in unreferenced:
            if not dry_run:
                path.unlink()
            logger.info(
                "%s unreferenced data file %s", "found" if dry_run else "deleted", path
            )
        return unreferenced


def _unreferenced(identity_dir: Path) -> list[Path]:
    """Return the files in ``identity_dir``'s ``data/`` named by no record file."""
    data_dir = identity_dir / "data"
    if not data_dir.is_dir():
        return []
    records_dir = identity_dir / "records"
    record_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(records_dir.glob("*.json"))
    )
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.name not in record_text
    )
