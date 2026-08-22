"""numpy support: scalars are stored inline in JSON, arrays as content-addressed .npy files."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import CorruptRecordError, SerializationError
from .base import DeserializationContext, SerializationContext
from .naming import content_filename

_SUPPORTED_SCALAR_KINDS = "iufbMmUS"


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
        content = obj.tobytes() + str(obj.shape).encode() + str(obj.dtype).encode()
        filename = content_filename("arr", content, "npy")

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
