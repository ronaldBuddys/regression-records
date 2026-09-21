"""polars support (optional dependency): DataFrame/Series are stored as parquet.

Files are content-addressed, and a Series is written as a one-column frame.

polars.LazyFrame is intentionally NOT auto-collected - collection can be expensive or
side-effecting, so we require the caller to materialize a DataFrame first.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl

from ..exceptions import CorruptRecordError, DeserializationError, SerializationError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename, verified_source


def _refuse_object_columns(frame: pl.DataFrame) -> None:
    """Raise ``SerializationError`` if any column holds arbitrary Python objects.

    ``Object`` is the one dtype parquet cannot hold, and polars says so with a bare
    ``ComputeError`` from deep inside the write. There is no pickle fallback as there
    is for pandas: every other polars dtype has a parquet form, so an ``Object`` column
    is a choice the caller can undo.
    """
    # Top-level only: polars refuses to construct an Object nested in a List or Struct.
    offending = [name for name, dtype in frame.schema.items() if dtype == pl.Object]
    if offending:
        raise SerializationError(
            f"polars column(s) {offending} have dtype Object, which parquet cannot "
            "hold. Cast them to a polars dtype (a Struct for records, String for "
            "labels) or drop them before returning/recording the frame."
        )


def _write_parquet(frame: pl.DataFrame, prefix: str, ctx: SerializationContext) -> str:
    """Write ``frame`` to a content-addressed parquet file and return its reference.

    The filename hashes the parquet bytes, so identical frames share one file.

    Raises ``SerializationError`` for a column of dtype ``Object``.
    """
    _refuse_object_columns(frame)
    buf = io.BytesIO()
    frame.write_parquet(buf)
    content = buf.getvalue()
    filename = content_filename(prefix, content, extension="parquet")

    def write(path: Path) -> None:
        path.write_bytes(content)

    ctx.write_if_absent(filename, write)
    return ctx.ref(filename)


def _read_parquet(blob: dict, ctx: DeserializationContext, what: str) -> pl.DataFrame:
    """Read the parquet file ``blob`` references, or raise ``CorruptRecordError``.

    Also raised when ``ctx.verify`` and the file does not match the hash in its name.
    """
    path = ctx.resolve(blob["path"])
    if not path.exists():
        raise CorruptRecordError(f"referenced polars {what} file is missing: {path}")
    return pl.read_parquet(verified_source(path, ctx.verify))


class PolarsDataFrameSerializer:
    """Stores ``polars.DataFrame`` values as content-addressed parquet side files."""

    type_tag = "polars.DataFrame"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``polars.DataFrame`` instances."""
        return isinstance(obj, pl.DataFrame)

    def store(self, obj: pl.DataFrame, ctx: SerializationContext) -> dict:
        """Write the frame to a content-addressed parquet file and return its reference.

        The filename hashes the parquet bytes, so identical frames share one file.
        """
        return {"path": _write_parquet(obj, "df", ctx), "shape": list(obj.shape)}

    def load(self, blob: dict, ctx: DeserializationContext) -> pl.DataFrame:
        """Read the frame from the parquet file the payload references.

        Raises ``CorruptRecordError`` if the file is missing, or fails ``ctx.verify``.
        """
        return _read_parquet(blob, ctx, "DataFrame")


class PolarsSeriesSerializer:
    """Stores ``polars.Series`` values as one-column parquet files, keeping the name."""

    type_tag = "polars.Series"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``polars.Series`` instances."""
        return isinstance(obj, pl.Series)

    def store(self, obj: pl.Series, ctx: SerializationContext) -> dict:
        """Write the series as a one-column frame and return its reference.

        The column carries the name, which for an unnamed polars series is ``""`` and
        round-trips as such. ``name``, ``length`` and ``dtype`` in the payload are for
        whoever reads the record; loading takes all three from the file.
        """
        return {
            "path": _write_parquet(obj.to_frame(), "series", ctx),
            "name": obj.name,
            "length": obj.len(),
            "dtype": str(obj.dtype),
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> pl.Series:
        """Read the series back from the one-column parquet file the payload references.

        Raises ``CorruptRecordError`` if the file is missing, or fails ``ctx.verify``.
        """
        return _read_parquet(blob, ctx, "Series").to_series()


class PolarsLazyFrameSerializer:
    """Claims ``polars.LazyFrame`` only to refuse it with a specific error."""

    type_tag = "polars.LazyFrame"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``polars.LazyFrame`` instances, keeping them from the fallback."""
        return isinstance(obj, pl.LazyFrame)

    def store(self, obj: pl.LazyFrame, ctx: SerializationContext) -> dict:
        """Raise ``SerializationError``: a LazyFrame must be collected first."""
        raise SerializationError(
            "polars.LazyFrame cannot be persisted as a regression record; call "
            "`.collect()` to materialize a DataFrame before returning/recording it. "
            "regression-records deliberately does not auto-collect LazyFrames, since "
            "collection can be expensive or have side effects."
        )

    def load(self, blob: dict, ctx: DeserializationContext) -> pl.LazyFrame:
        """Raise ``DeserializationError``: no LazyFrame record is ever written."""
        raise DeserializationError("polars.LazyFrame records are not supported")
