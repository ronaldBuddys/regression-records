"""Convert arbitrary Python objects to/from a JSON-safe tree, offloading inconvenient
types (arrays, dataframes, arbitrary objects) to external files via a
:class:`~regression_records.serialization.base.SerializerRegistry`.

The public entry points are :func:`to_json_tree` / :func:`from_json_tree`, plus
:func:`register_serializer` for extending the module-level :data:`default_registry`
with support for additional types.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import SerializationError
from .base import (
    DeserializationContext,
    SerializationContext,
    Serializer,
    SerializerRegistry,
)

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


def _delegate(obj: Any, ctx: SerializationContext) -> dict:
    """Hand ``obj`` to the serializer that claims it, tagging the payload it returns."""
    serializer = ctx.registry.find(obj)
    if serializer is None:
        raise SerializationError(
            f"no serializer available for object of type "
            f"'{type(obj).__module__}.{type(obj).__qualname__}'; register a custom "
            "Serializer via regression_records.serialization.register_serializer()"
        )
    return {TYPE_KEY: serializer.type_tag, **serializer.store(obj, ctx)}


def to_json_tree(obj: Any, ctx: SerializationContext) -> Any:
    """Recursively convert ``obj`` into a JSON-safe structure.

    Only the values JSON represents faithfully - primitives, lists, and string-keyed
    dicts - are handled here; everything else is delegated to ``ctx.registry`` (writing
    external files under ``ctx.data_dir`` as needed). Raises :class:`SerializationError`
    if no serializer can handle a value.
    """
    # numpy scalar types (e.g. np.float64) subclass Python's float/int/bool, so they
    # have to reach the registry before the primitive check below, or their dtype would
    # be silently flattened away into a plain Python number.
    #
    # Testing exact types below instead would make this special case unnecessary, but it
    # would also reroute every *other* builtin subclass - IntEnum, StrEnum, anything
    # subclassing int for units - to the generic object serializer, where an Enum member
    # fails to record at all. Recording those as their underlying builtin is right, and
    # ``isinstance`` is what says so.
    if isinstance(obj, np.generic):
        return _delegate(obj, ctx)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, dict):
        return _dict_to_json_tree(obj, ctx)

    if isinstance(obj, list):
        return [to_json_tree(v, ctx) for v in obj]

    return _delegate(obj, ctx)


def from_json_tree(tree: Any, ctx: DeserializationContext) -> Any:
    """Reconstruct the Python object previously converted by :func:`to_json_tree`."""
    if isinstance(tree, dict):
        if TYPE_KEY in tree:
            return ctx.registry.by_tag(tree[TYPE_KEY]).load(tree, ctx)
        return {k: from_json_tree(v, ctx) for k, v in tree.items()}
    if isinstance(tree, list):
        return [from_json_tree(v, ctx) for v in tree]
    return tree


def _build_default_registry() -> SerializerRegistry:
    registry = SerializerRegistry()

    from .numpy_serializer import NumpyArraySerializer, NumpyScalarSerializer
    from .primitives import (
        BytesSerializer,
        DateSerializer,
        DatetimeSerializer,
        SetSerializer,
        TupleSerializer,
    )

    registry.register(NumpyScalarSerializer())
    registry.register(NumpyArraySerializer())
    registry.register(DatetimeSerializer())
    registry.register(DateSerializer())
    registry.register(BytesSerializer())
    registry.register(TupleSerializer())
    registry.register(SetSerializer())

    try:
        from .polars_serializer import (
            PolarsDataFrameSerializer,
            PolarsLazyFrameSerializer,
        )
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

    # Catch-all for arbitrary objects. Registered as the fallback so it keeps last
    # refusal no matter what is registered afterwards.
    from .object_serializer import GenericObjectSerializer

    registry.register(GenericObjectSerializer(), fallback=True)

    return registry


default_registry = _build_default_registry()


def register_serializer(serializer: Serializer, *, priority: int | None = None) -> None:
    """Register ``serializer`` on the module-level :data:`default_registry`.

    The serializer is inserted just before the generic object fallback, so it takes
    priority over that but not over the more specific built-in serializers - pass an
    explicit ``priority`` index to override even those.
    """
    default_registry.register(serializer, priority=priority)
