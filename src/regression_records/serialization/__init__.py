"""Convert arbitrary Python objects to/from a JSON-safe tree, offloading inconvenient
types (arrays, dataframes, arbitrary objects) to external files via a
:class:`~regression_records.serialization.base.SerializerRegistry`.

The public entry points are :func:`to_json_tree` / :func:`from_json_tree`, plus
:func:`register_serializer` for extending the module-level :data:`default_registry`
with support for additional types.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..exceptions import SerializationError
from .base import DeserializationContext, SerializationContext, Serializer, SerializerRegistry

__all__ = [
    "DeserializationContext",
    "SerializationContext",
    "Serializer",
    "SerializerRegistry",
    "default_registry",
    "from_json_tree",
    "register_serializer",
    "to_json_tree",
]

# Reserved key used to tag non-primitive values in the JSON tree. A plain dict being
# recorded must not itself use this key (checked and rejected with a clear error).
TYPE_KEY = "__type__"


def _dict_to_json_tree(obj: dict, ctx: SerializationContext) -> dict:
    if TYPE_KEY in obj:
        raise SerializationError(
            f"dict contains the key {TYPE_KEY!r}, which regression-records reserves "
            "for its internal serialization format; rename this key before recording"
        )
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            raise SerializationError(
                f"dict keys must be strings to be recorded, got key {key!r} of type "
                f"{type(key).__name__}; convert keys to strings before recording"
            )
        out[key] = to_json_tree(value, ctx)
    return out


def to_json_tree(obj: Any, ctx: SerializationContext) -> Any:
    """Recursively convert ``obj`` into a JSON-safe structure.

    Dicts/lists/tuples/sets/primitives are handled structurally; anything else is
    delegated to ``ctx.registry`` (writing external files under ``ctx.data_dir`` as
    needed). Raises :class:`SerializationError` if no serializer can handle a value.
    """
    # numpy scalar types (e.g. np.float64) subclass Python's float/int/bool, so this
    # check must run before the primitive check below to preserve dtype information.
    if isinstance(obj, np.generic):
        serializer = ctx.registry.find(obj)
        if serializer is None:  # pragma: no cover - default registry always has one
            raise SerializationError("no serializer registered for numpy scalars")
        payload = serializer.store(obj, ctx)
        return {TYPE_KEY: serializer.type_tag, **payload}

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, bytes):
        return {TYPE_KEY: "bytes", "hex": obj.hex()}

    if isinstance(obj, dict):
        return _dict_to_json_tree(obj, ctx)

    if isinstance(obj, list):
        return [to_json_tree(v, ctx) for v in obj]

    if isinstance(obj, tuple):
        return {TYPE_KEY: "tuple", "items": [to_json_tree(v, ctx) for v in obj]}

    if isinstance(obj, set):
        items = [to_json_tree(v, ctx) for v in obj]
        # Set iteration order isn't stable across processes; sort deterministically so
        # identical sets always hash/serialize identically.
        items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return {TYPE_KEY: "set", "items": items}

    serializer = ctx.registry.find(obj)
    if serializer is None:
        raise SerializationError(
            f"no serializer available for object of type "
            f"'{type(obj).__module__}.{type(obj).__qualname__}'; register a custom "
            "Serializer via regression_records.serialization.register_serializer()"
        )
    payload = serializer.store(obj, ctx)
    return {TYPE_KEY: serializer.type_tag, **payload}


def from_json_tree(tree: Any, ctx: DeserializationContext) -> Any:
    """Reconstruct the Python object previously converted by :func:`to_json_tree`."""
    if isinstance(tree, dict):
        if TYPE_KEY in tree:
            tag = tree[TYPE_KEY]
            if tag == "tuple":
                return tuple(from_json_tree(v, ctx) for v in tree["items"])
            if tag == "set":
                return {from_json_tree(v, ctx) for v in tree["items"]}
            if tag == "bytes":
                return bytes.fromhex(tree["hex"])
            serializer = ctx.registry.by_tag(tag)
            return serializer.load(tree, ctx)
        return {k: from_json_tree(v, ctx) for k, v in tree.items()}
    if isinstance(tree, list):
        return [from_json_tree(v, ctx) for v in tree]
    return tree


def _build_default_registry() -> SerializerRegistry:
    registry = SerializerRegistry()

    from .numpy_serializer import NumpyArraySerializer, NumpyScalarSerializer
    from .primitives import DateSerializer, DatetimeSerializer

    registry.register(NumpyScalarSerializer())
    registry.register(NumpyArraySerializer())
    registry.register(DatetimeSerializer())
    registry.register(DateSerializer())

    try:
        from .polars_serializer import PolarsDataFrameSerializer, PolarsLazyFrameSerializer
    except ImportError:
        pass
    else:
        registry.register(PolarsDataFrameSerializer())
        registry.register(PolarsLazyFrameSerializer())

    try:
        from .pandas_serializer import PandasDataFrameSerializer, PandasSeriesSerializer
    except ImportError:
        pass
    else:
        registry.register(PandasDataFrameSerializer())
        registry.register(PandasSeriesSerializer())

    # Fallback for arbitrary objects - must stay last so more specific serializers
    # above always get first refusal.
    from .object_serializer import GenericObjectSerializer

    registry.register(GenericObjectSerializer())

    return registry


default_registry = _build_default_registry()


def register_serializer(serializer: Serializer, *, priority: int | None = None) -> None:
    """Register ``serializer`` on the module-level :data:`default_registry`.

    By default the serializer is inserted just before the generic object fallback, so
    it takes priority over it but not over more specific built-in serializers unless
    an explicit ``priority`` index is given.
    """
    if priority is None:
        priority = max(len(default_registry) - 1, 0)
    default_registry.register(serializer, priority=priority)
