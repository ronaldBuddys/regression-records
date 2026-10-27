"""pandas support (optional dependency): DataFrame/Series are stored as parquet.

Parquet cannot hold every frame pandas can (an object column of mixed types, for one).
Such a frame is refused unless the serializer was built with ``allow_pickle=True``, in
which case that one file is pickled instead. Pickle is opt-in, in both directions,
because unpickling runs code: a store is committed and shared, so reading a pickle
must be something the *reader* chose, never something a record file can ask for.
"""

from __future__ import annotations

import io
import pickle
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import CorruptRecordError, DeserializationError, SerializationError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename, verified_source

_MISSING_SERIES_NAME = "__regression_records_series__"

_OPT_IN = (
    "register PandasDataFrameSerializer(allow_pickle=True) / "
    "PandasSeriesSerializer(allow_pickle=True) with replace=True on a copy of the "
    "default registry"
)


def _write_parquet_or_pickle(
    frame: pd.DataFrame, prefix: str, ctx: SerializationContext, *, allow_pickle: bool
) -> dict:
    try:
        buf = io.BytesIO()
        frame.to_parquet(buf)
        content = buf.getvalue()
        fmt = "parquet"
        extension = "parquet"
    except ImportError as e:
        # Never a reason to pickle: without a parquet engine *every* frame would fail
        # here, and the whole store would quietly become pickles.
        raise SerializationError(
            "recording a pandas object needs a parquet engine; install "
            f"regression-records[pandas], which brings pyarrow ({e})"
        ) from e
    except Exception as e:  # noqa: BLE001 - engines differ in what they raise
        if not allow_pickle:
            raise SerializationError(
                f"cannot store this pandas object as parquet ({e!r}). Convert the "
                "offending columns to dtypes Arrow supports, or accept a pickled "
                f"side file for such frames: {_OPT_IN}."
            ) from e
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

    def write(path: Path) -> None:
        path.write_bytes(content)

    ctx.write_if_absent(filename, write)
    return {"path": ctx.ref(filename), "format": fmt}


def _read_parquet_or_pickle(
    blob: dict, ctx: DeserializationContext, *, allow_pickle: bool
) -> pd.DataFrame:
    path = ctx.resolve(blob["path"])
    if not path.exists():
        raise CorruptRecordError(f"referenced pandas data file is missing: {path}")
    if blob["format"] == "pickle":
        if not allow_pickle:
            # The format is a field of the record file, so without this check editing
            # a record would be enough to have an arbitrary file unpickled.
            raise DeserializationError(
                f"record references a pickled pandas object ({path.name}), and "
                f"unpickling runs code. If you trust this store, {_OPT_IN}."
            )
        # With verify_side_files the bytes are checked before unpickling, so a swapped
        # file is refused rather than run; without it, the opt-in above is the only gate.
        # Pickle is deliberate here: reached only with allow_pickle=True, see above.
        return pd.read_pickle(verified_source(path, ctx.verify))  # noqa: S301
    return pd.read_parquet(verified_source(path, ctx.verify))


class PandasDataFrameSerializer:
    """Stores ``pandas.DataFrame`` values as content-addressed parquet side files."""

    type_tag = "pandas.DataFrame"

    def __init__(self, *, allow_pickle: bool = False) -> None:
        #: Whether a frame parquet cannot hold may be pickled, and a pickle read back.
        self.allow_pickle = allow_pickle

    def can_handle(self, obj: Any) -> bool:
        """Claim ``pandas.DataFrame`` instances."""
        return isinstance(obj, pd.DataFrame)

    def store(self, obj: pd.DataFrame, ctx: SerializationContext) -> dict:
        """Write the frame to a content-addressed parquet file and return its reference.

        When parquet cannot hold the frame this raises ``SerializationError``, or, with
        ``allow_pickle=True``, writes a pickle file and warns; the payload's ``format``
        says which was written.
        """
        payload = _write_parquet_or_pickle(
            obj, "pddf", ctx, allow_pickle=self.allow_pickle
        )
        payload["shape"] = list(obj.shape)
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> pd.DataFrame:
        """Read the frame back from the parquet (or pickle) file the payload references.

        Raises ``CorruptRecordError`` if the file is missing or fails ``ctx.verify``, and
        ``DeserializationError`` for a pickle unless ``allow_pickle=True``.
        """
        return _read_parquet_or_pickle(blob, ctx, allow_pickle=self.allow_pickle)


class PandasSeriesSerializer:
    """Stores ``pandas.Series`` values as one-column parquet files, keeping the name."""

    type_tag = "pandas.Series"

    def __init__(self, *, allow_pickle: bool = False) -> None:
        #: Whether a frame parquet cannot hold may be pickled, and a pickle read back.
        self.allow_pickle = allow_pickle

    def can_handle(self, obj: Any) -> bool:
        """Claim ``pandas.Series`` instances."""
        return isinstance(obj, pd.Series)

    def store(self, obj: pd.Series, ctx: SerializationContext) -> dict:
        """Write the series as a one-column frame and return its reference.

        An unnamed series is given a placeholder column name for the file; the original
        name travels in the payload. A series parquet cannot hold is refused or
        pickled exactly as the dataframe serializer does.
        """
        from . import to_json_tree

        # Parquet column names must be strings, and a series name often is not: an
        # int after a groupby, a tuple from a MultiIndex level. Those are written
        # under the placeholder too, and the real name travels in the payload.
        series_name = obj.name if isinstance(obj.name, str) else _MISSING_SERIES_NAME
        frame = obj.rename(series_name).to_frame()
        payload = _write_parquet_or_pickle(
            frame, "pdseries", ctx, allow_pickle=self.allow_pickle
        )
        payload["series_name"] = series_name
        # Through the tree walker, so that a numpy scalar or tuple name is JSON-safe
        # rather than failing later, in json.dump, far from its cause.
        payload["original_name"] = to_json_tree(obj.name, ctx)
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> pd.Series:
        """Read the stored column back and restore the series' original name.

        Raises ``CorruptRecordError`` if the file is missing or fails ``ctx.verify``, and
        ``DeserializationError`` for a pickle unless ``allow_pickle=True``.
        """
        frame = _read_parquet_or_pickle(blob, ctx, allow_pickle=self.allow_pickle)
        from . import from_json_tree

        series = frame[blob["series_name"]]
        return series.rename(from_json_tree(blob["original_name"], ctx))


class PandasTimestampSerializer:
    """Stores ``pandas.Timestamp`` (and ``NaT``) to the nanosecond, keeping the type.

    A ``Timestamp`` is a ``datetime`` subclass, and was recorded as a plain datetime -
    losing nanoseconds, and failing every later comparison as a type mismatch. Must be
    registered ahead of the datetime serializer. The zone is kept by name when it has
    one (``ZoneInfo`` or pytz); any other comes back as its fixed UTC offset.
    """

    type_tag = "pandas.Timestamp"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``pandas.Timestamp`` instances and ``pandas.NaT``."""
        return isinstance(obj, pd.Timestamp) or obj is pd.NaT

    def store(self, obj: pd.Timestamp, ctx: SerializationContext) -> dict:
        """Return the ISO 8601 string to the nanosecond, plus the zone name if any."""
        payload: dict[str, Any] = {"value": obj.isoformat()}
        tz_name = getattr(obj.tz, "key", None) or getattr(obj.tz, "zone", None)
        if tz_name is not None:
            payload["tz"] = tz_name
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> pd.Timestamp:
        """Parse the stored string, converting back to the recorded zone."""
        value = pd.Timestamp(blob["value"])
        if "tz" in blob:
            value = value.tz_convert(blob["tz"])
        return value


class PandasTimedeltaSerializer:
    """Stores ``pandas.Timedelta`` as integer nanoseconds, keeping the type.

    A ``Timedelta`` is a ``datetime.timedelta`` subclass; must be registered ahead of
    that serializer, which would drop the nanoseconds and the type.
    """

    type_tag = "pandas.Timedelta"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``pandas.Timedelta`` instances."""
        return isinstance(obj, pd.Timedelta)

    def store(self, obj: pd.Timedelta, ctx: SerializationContext) -> dict:
        """Return the duration in integer nanoseconds."""
        return {"nanoseconds": obj.value}

    def load(self, blob: dict, ctx: DeserializationContext) -> pd.Timedelta:
        """Rebuild the timedelta from its nanoseconds."""
        return pd.Timedelta(blob["nanoseconds"], unit="ns")
