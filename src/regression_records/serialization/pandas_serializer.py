"""pandas support (optional dependency): DataFrame/Series are stored as parquet by default.

If parquet storage fails (e.g. object columns with mixed, Arrow-incompatible types),
storage falls back to pickle for that single file - a narrow, documented exception to
the "avoid pickle" rule, matching what pandas itself can and cannot round-trip safely.
"""

from __future__ import annotations

import io
import pickle
import warnings
from typing import Any

import pandas as pd

from ..exceptions import CorruptRecordError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename

_MISSING_SERIES_NAME = "__regression_records_series__"


def _write_parquet_or_pickle(frame: pd.DataFrame, prefix: str, ctx: SerializationContext) -> dict:
    try:
        buf = io.BytesIO()
        frame.to_parquet(buf)
        content = buf.getvalue()
        fmt = "parquet"
        extension = "parquet"
    except Exception as e:
        warnings.warn(
            f"failed to store pandas object as parquet ({e!r}); falling back to pickle. "
            "This reduces portability/inspectability - consider using dtypes supported "
            "by Arrow if this recurs.",
            stacklevel=3,
        )
        content = pickle.dumps(frame)
        fmt = "pickle"
        extension = "pkl"

    filename = content_filename(prefix, content, extension=extension)

    def write(path):
        path.write_bytes(content)

    ctx.write_if_absent(filename, write)
    return {"path": ctx.ref(filename), "format": fmt}


def _read_parquet_or_pickle(blob: dict, ctx: DeserializationContext) -> pd.DataFrame:
    path = ctx.resolve(blob["path"])
    if not path.exists():
        raise CorruptRecordError(f"referenced pandas data file is missing: {path}")
    if blob["format"] == "pickle":
        return pd.read_pickle(path)
    return pd.read_parquet(path)


class PandasDataFrameSerializer:
    type_tag = "pandas.DataFrame"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, pd.DataFrame)

    def store(self, obj: pd.DataFrame, ctx: SerializationContext) -> dict:
        payload = _write_parquet_or_pickle(obj, "pddf", ctx)
        payload["shape"] = list(obj.shape)
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> pd.DataFrame:
        return _read_parquet_or_pickle(blob, ctx)


class PandasSeriesSerializer:
    type_tag = "pandas.Series"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, pd.Series)

    def store(self, obj: pd.Series, ctx: SerializationContext) -> dict:
        series_name = obj.name if obj.name is not None else _MISSING_SERIES_NAME
        frame = obj.rename(series_name).to_frame()
        payload = _write_parquet_or_pickle(frame, "pdseries", ctx)
        payload["series_name"] = series_name
        payload["original_name"] = obj.name
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> pd.Series:
        frame = _read_parquet_or_pickle(blob, ctx)
        series = frame[blob["series_name"]]
        return series.rename(blob["original_name"])
