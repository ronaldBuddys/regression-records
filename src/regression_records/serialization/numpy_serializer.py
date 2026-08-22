"""numpy support: scalars are stored inline in JSON, arrays as content-addressed .npy files."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import CorruptRecordError, SerializationError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename

_SUPPORTED_SCALAR_KINDS = "iufbMmUS"


def _raw_bytes(obj: np.ndarray) -> memoryview:
    """A read-only view of ``obj``'s elements, for hashing without copying them.

    Equivalent to ``obj.tobytes()`` byte for byte - both produce C-ordered element
    bytes - but a view rather than a copy whenever the array is already contiguous.
    That matters because every argument of every recorded call is hashed, so a copy
    here is a copy of the whole array on every call, not only when one is written.

    The ``uint8`` view is what makes it work for every dtype: ``datetime64`` and
    ``timedelta64`` refuse to expose a buffer directly.
    """
    return np.ascontiguousarray(obj).view(np.uint8).reshape(-1).data


class NumpyScalarSerializer:
    type_tag = "numpy.scalar"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, np.generic)

    def store(self, obj: np.generic, ctx: SerializationContext) -> dict:
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
        dtype = np.dtype(blob["dtype"])
        value = blob["value"]
        if dtype.kind == "M":
            return np.datetime64(value).astype(dtype)
        if dtype.kind == "m":
            return np.timedelta64(value, "ns").astype(dtype)
        return dtype.type(value)


class NumpyArraySerializer:
    type_tag = "numpy.ndarray"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def store(self, obj: np.ndarray, ctx: SerializationContext) -> dict:
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
        return {"path": ctx.ref(filename), "shape": list(obj.shape), "dtype": str(obj.dtype)}

    def load(self, blob: dict, ctx: DeserializationContext) -> np.ndarray:
        path = ctx.resolve(blob["path"])
        if not path.exists():
            raise CorruptRecordError(f"referenced numpy array file is missing: {path}")
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as e:
            raise CorruptRecordError(f"could not load numpy array file {path}: {e}") from e
