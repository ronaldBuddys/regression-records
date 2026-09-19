"""polars support (optional dependency): DataFrames are stored as content-addressed parquet.

polars.LazyFrame is intentionally NOT auto-collected - collection can be expensive or
side-effecting, so we require the caller to materialize a DataFrame first.
"""

from __future__ import annotations

import io
from typing import Any

import polars as pl

from ..exceptions import CorruptRecordError, DeserializationError, SerializationError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename


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
        buf = io.BytesIO()
        obj.write_parquet(buf)
        content = buf.getvalue()
        filename = content_filename("df", content, extension="parquet")

        def write(path):
            path.write_bytes(content)

        ctx.write_if_absent(filename, write)
        return {"path": ctx.ref(filename), "shape": list(obj.shape)}

    def load(self, blob: dict, ctx: DeserializationContext) -> pl.DataFrame:
        """Read the frame from the parquet file the payload references.

        Raises ``CorruptRecordError`` if the file is missing.
        """
        path = ctx.resolve(blob["path"])
        if not path.exists():
            raise CorruptRecordError(
                f"referenced polars DataFrame file is missing: {path}"
            )
        return pl.read_parquet(path)


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
