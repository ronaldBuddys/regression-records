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
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .exceptions import CorruptRecordError
from .identity import FunctionIdentity
from .records import RegressionRecord
from .serialization import default_registry as _default_registry
from .serialization.base import DeserializationContext, SerializationContext, SerializerRegistry


#: How much of each hex identifier goes into a record's filename. The input hash leads
#: so that looking a case up is a glob rather than a scan; the record id follows to keep
#: distinct recordings of the same inputs (e.g. after an intentional behavior change)
#: from colliding.
_HASH_CHARS = 16
_ID_CHARS = 8


def record_filename_prefix(input_hash: str) -> str:
    """The shared filename prefix of every record with this input hash."""
    return f"rec-{input_hash[:_HASH_CHARS]}-"


def record_filename(input_hash: str, record_id: str) -> str:
    """The on-disk filename for a record with these identifiers."""
    return f"{record_filename_prefix(input_hash)}{record_id[:_ID_CHARS]}.json"


class RecordStore:
    """Reads and writes :class:`~regression_records.records.RegressionRecord`\\ s on disk."""

    def __init__(self, root: str | Path, *, serializer_registry: Optional[SerializerRegistry] = None):
        self.root = Path(root)
        self.registry = serializer_registry or _default_registry

    def _identity_dir(self, identity: FunctionIdentity) -> Path:
        return self.root.joinpath(*identity.path_parts)

    def _records_dir(self, identity: FunctionIdentity) -> Path:
        return self._identity_dir(identity) / "records"

    def _data_dir(self, identity: FunctionIdentity) -> Path:
        return self._identity_dir(identity) / "data"

    def serialization_context(
        self, identity: FunctionIdentity, *, deferred: bool = False
    ) -> SerializationContext:
        # data/ is a sibling of records/, so references stored in a record's JSON must
        # be prefixed to resolve correctly relative to the records/ directory.
        return SerializationContext(
            data_dir=self._data_dir(identity),
            registry=self.registry,
            ref_prefix="../data/",
            deferred=deferred,
        )

    def deserialization_context(self, identity: FunctionIdentity) -> DeserializationContext:
        # External files are referenced relative to records/ (e.g. "../data/arr-....npy").
        return DeserializationContext(base_dir=self._records_dir(identity), registry=self.registry)

    def _record_path(self, identity: FunctionIdentity, record: RegressionRecord) -> Path:
        return self._records_dir(identity) / record_filename(record.input_hash, record.record_id)

    def save(self, record: RegressionRecord, *, overwrite: bool = False) -> Path:
        """Write ``record`` to disk. Raises ``FileExistsError`` unless ``overwrite=True``."""
        records_dir = self._records_dir(record.identity)
        records_dir.mkdir(parents=True, exist_ok=True)
        path = self._record_path(record.identity, record)
        if path.exists() and not overwrite:
            raise FileExistsError(f"record file already exists: {path}")
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, sort_keys=True)
        return path

    def _load_record_file(self, path: Path) -> RegressionRecord:
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise CorruptRecordError(f"could not read record file {path}: {e}") from e
        try:
            return RegressionRecord.from_dict(raw)
        except (KeyError, TypeError) as e:
            raise CorruptRecordError(f"record file {path} is malformed: {e}") from e

    def list_records(self, identity: FunctionIdentity) -> list[RegressionRecord]:
        """Return all historical records for ``identity``, oldest-filename-first."""
        records_dir = self._records_dir(identity)
        if not records_dir.exists():
            return []
        return [self._load_record_file(path) for path in sorted(records_dir.glob("rec-*.json"))]

    def find_by_input_hash(
        self, identity: FunctionIdentity, input_hash: str
    ) -> Optional[RegressionRecord]:
        """Return the record for ``identity`` whose recorded inputs match exactly.

        Because the input hash is part of the filename, this is a directory glob rather
        than a scan of every historical record - it stays cheap as a function
        accumulates recorded cases, and it runs on every validated call.
        """
        records_dir = self._records_dir(identity)
        if not records_dir.exists():
            return None
        for path in sorted(records_dir.glob(f"{record_filename_prefix(input_hash)}*.json")):
            record = self._load_record_file(path)
            # The filename holds a truncated hash; confirm against the full one inside.
            if record.input_hash == input_hash:
                return record
        return None

    def has_records(self, identity: FunctionIdentity) -> bool:
        """Whether any record exists for ``identity``, without parsing any of them."""
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
        """
        old_dir = self._identity_dir(old)
        new_dir = self._identity_dir(new)
        if not old_dir.exists():
            raise FileNotFoundError(f"no records exist for identity at {old_dir}")
        if new_dir.exists():
            raise FileExistsError(f"records already exist for the new identity at {new_dir}")
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)

        for path in sorted((new_dir / "records").glob("rec-*.json")):
            record = self._load_record_file(path)
            record.identity = replace(
                record.identity, module=new.module, qualname=new.qualname
            )
            with open(path, "w") as f:
                json.dump(record.to_dict(), f, indent=2, sort_keys=True)
