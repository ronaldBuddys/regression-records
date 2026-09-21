"""numpy support: scalars are stored inline in JSON, arrays as content-addressed .npy files."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import CorruptRecordError, SerializationError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename

_SUPPORTED_SCALAR_KINDS = "iufbMmUS"


def _raw_bytes(obj: np.ndarray) -> memoryview:
    """Return a read-only view of ``obj``'s elements, for hashing without copying them.

    Equivalent to ``obj.tobytes()`` byte for byte - both produce C-ordered element
    bytes - but a view rather than a copy whenever the array is already contiguous.
    That matters because every argument of every recorded call is hashed, so a copy
    here is a copy of the whole array on every call, not only when one is written.

    The ``uint8`` view is what makes it work for every dtype: ``datetime64`` and
    ``timedelta64`` refuse to expose a buffer directly.
    """
    return np.ascontiguousarray(obj).view(np.uint8).reshape(-1).data


class NumpyScalarSerializer:
    """Stores numpy scalars inline in the record's JSON, keeping their dtype."""

    type_tag = "numpy.scalar"

    def can_handle(self, obj: Any) -> bool:
        """Claim numpy scalars (``np.float64``, ``np.datetime64``, ...)."""
        return isinstance(obj, np.generic)

    def store(self, obj: np.generic, ctx: SerializationContext) -> dict:
        """Return the scalar's dtype and a JSON-safe value; no side file is written.

        Datetimes are stored as ISO strings and timedeltas as integer nanoseconds.
        Raises ``SerializationError`` for dtype kinds outside bool, int, float,
        datetime, timedelta and string (e.g. complex).
        """
        dtype = obj.dtype
        if dtype.kind not in _SUPPORTED_SCALAR_KINDS:
            raise SerializationError(
                f"unsupported numpy scalar dtype: {dtype!r} (kind {dtype.kind!r})"
            )
        if dtype.kind == "M":
            value: Any = str(obj)
        elif dtype.kind == "m":
            value = int(obj.astype("timedelta64[ns]").astype(np.int64))
        elif dtype.kind == "b":
            value = bool(obj)
        elif dtype.kind in "US":
            value = str(obj)
        else:
            value = obj.item()
        return {"dtype": str(dtype), "value": value}

    def load(self, blob: dict, ctx: DeserializationContext) -> np.generic:
        """Rebuild the scalar from its inline value, restoring the recorded dtype."""
        dtype = np.dtype(blob["dtype"])
        value = blob["value"]
        if dtype.kind == "M":
            return np.datetime64(value).astype(dtype)
        if dtype.kind == "m":
            return np.timedelta64(value, "ns").astype(dtype)
        return dtype.type(value)


class NumpyArraySerializer:
    """Stores ``numpy.ndarray`` values as content-addressed ``.npy`` side files."""

    type_tag = "numpy.ndarray"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``numpy.ndarray`` instances, subclasses included."""
        return isinstance(obj, np.ndarray)

    def store(self, obj: np.ndarray, ctx: SerializationContext) -> dict:
        """Save the array as a content-addressed ``.npy`` file and return its reference.

        The filename hashes the elements, shape and dtype, so identical arrays share
        one file. Raises ``SerializationError`` for ``dtype=object`` arrays, which would
        need pickling.
        """
        if obj.dtype == np.dtype("O"):
            raise SerializationError(
                "numpy arrays with dtype=object are not supported (not portable and "
                "requires unpickling to load); convert to a fixed dtype or a plain "
                "list before recording"
            )
        filename = content_filename(
            "arr",
            _raw_bytes(obj),
            str(obj.shape).encode(),
            str(obj.dtype).encode(),
            extension="npy",
        )

        def write(path):
            np.save(path, obj, allow_pickle=False)

        ctx.write_if_absent(filename, write)
        return {
            "path": ctx.ref(filename),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> np.ndarray:
        """Load the array from the ``.npy`` file the payload references, without pickle.

        Raises ``CorruptRecordError`` if the file is missing or unreadable.
        """
        path = ctx.resolve(blob["path"])
        if not path.exists():
            raise CorruptRecordError(f"referenced numpy array file is missing: {path}")
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as e:
            raise CorruptRecordError(
                f"could not load numpy array file {path}: {e}"
            ) from e
