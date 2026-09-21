"""Convert arbitrary Python objects to and from a JSON-safe tree.

Inconvenient types (arrays, dataframes, arbitrary objects) are offloaded to external
files via a :class:`~regression_records.serialization.base.SerializerRegistry`.

The public entry points are :func:`to_json_tree` / :func:`from_json_tree`, plus
:func:`register_serializer` for extending the module-level :data:`default_registry`
with support for additional types.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .._optional import library_is_absent
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


def _list_to_json_tree(obj: list, ctx: SerializationContext) -> list:
    return [to_json_tree(v, ctx) for v in obj]


def _delegate(obj: Any, ctx: SerializationContext) -> dict:
    """Hand ``obj`` to the serializer that claims it, tagging the payload it returns."""
    serializer = ctx.registry.find(obj)
    if serializer is None:
        raise SerializationError(
            f"no serializer available for object of type "
            f"'{type(obj).__module__}.{type(obj).__qualname__}'; register a custom "
            "Serializer via regression_records.serialization.register_serializer()"
        )
    payload = serializer.store(obj, ctx)
    if TYPE_KEY in payload:
        # The tag decides which serializer reads the value back, so a payload must not
        # be able to replace it - the same rule a recorded dict is held to.
        raise SerializationError(
            f"{type(serializer).__name__}.store() returned the reserved key "
            f"{TYPE_KEY!r}; a serializer names its type through type_tag"
        )
    return {TYPE_KEY: serializer.type_tag, **payload}


def _descend(
    obj: Any,
    ctx: SerializationContext,
    convert: Callable[[Any, SerializationContext], Any],
) -> Any:
    """Convert a container, refusing one that is already being converted.

    Only containers *in progress* are tracked, so an object referenced twice is fine
    (it is simply recorded twice); one that contains itself is not, and would otherwise
    recurse until ``RecursionError`` escaped from the decorated call.
    """
    key = id(obj)
    if key in ctx.active:
        raise SerializationError(
            f"cannot record a '{type(obj).__qualname__}' that contains a reference "
            "back to itself (a parent/child link, for instance): a record is a tree. "
            "With record_method, leave the back-reference out with exclude_attrs=(...); "
            "otherwise write a custom Serializer for the type."
        )
    ctx.active.add(key)
    try:
        return convert(obj, ctx)
    finally:
        ctx.active.discard(key)


def to_json_tree(obj: Any, ctx: SerializationContext) -> Any:
    """Recursively convert ``obj`` into a JSON-safe structure.

    Only the values JSON represents faithfully - primitives, lists, and string-keyed
    dicts - are handled here; everything else is delegated to ``ctx.registry`` (writing
    external files under ``ctx.data_dir`` as needed). Raises :class:`SerializationError`
    if no serializer can handle a value.

    Parameters
    ----------
    obj : Any
        The value to convert. A ``bool``, ``int``, ``float`` or ``str`` subclass
        (an ``IntEnum``, say) is recorded as its underlying builtin; a numpy scalar
        keeps its dtype.
    ctx : SerializationContext
        Supplies the serializer registry and the directory side files go to.

    Returns
    -------
    Any
        ``None``, a ``bool``, ``int``, ``float`` or ``str``, or a ``list`` / ``dict``
        of such values. A delegated value becomes a dict tagged with the reserved key
        ``"__type__"``, which :func:`from_json_tree` dispatches on.

    Raises
    ------
    SerializationError
        If no serializer claims a value; a dict has a non-string key or uses the
        reserved key ``"__type__"``; a serializer's payload uses that key; a
        container holds a reference back to itself; or the serializer chosen refuses
        the value.

    Notes
    -----
    A NaN or infinite float is tagged (``{"__type__": "float", "value": "nan"}``)
    rather than passed through, so the tree is strict JSON (RFC 8259), not just what
    Python's ``json`` module accepts.
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
    #
    # Non-finite floats go to the registry too: JSON has no literal for them.
    if isinstance(obj, np.generic) or (
        isinstance(obj, float) and not math.isfinite(obj)
    ):
        return _delegate(obj, ctx)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, dict):
        return _descend(obj, ctx, _dict_to_json_tree)

    if isinstance(obj, list):
        return _descend(obj, ctx, _list_to_json_tree)

    return _descend(obj, ctx, _delegate)


def from_json_tree(tree: Any, ctx: DeserializationContext) -> Any:
    """Reconstruct the Python object previously converted by :func:`to_json_tree`.

    Parameters
    ----------
    tree : Any
        A tree produced by :func:`to_json_tree`, typically read back from a record.
    ctx : DeserializationContext
        Supplies the serializer registry and resolves side-file references.

    Returns
    -------
    Any
        The reconstructed value. Untagged dicts and lists are rebuilt as new
        containers; primitives are returned as they are.

    Raises
    ------
    DeserializationError
        If a tagged value names a type tag with no registered serializer, or the
        serializer it names cannot load it. :class:`CorruptRecordError`, for a missing
        or out-of-store side file, is a subclass.
    IdentityResolutionError
        If a recorded object's class can no longer be imported by name.
    """
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
        NamedTupleSerializer,
        NonFiniteFloatSerializer,
        SetSerializer,
        TupleSerializer,
    )

    registry.register(NumpyScalarSerializer())
    registry.register(NumpyArraySerializer())
    registry.register(DatetimeSerializer())
    registry.register(DateSerializer())
    registry.register(NonFiniteFloatSerializer())
    registry.register(BytesSerializer())
    # Ahead of TupleSerializer, which would otherwise claim a named tuple first.
    registry.register(NamedTupleSerializer())
    registry.register(TupleSerializer())
    registry.register(SetSerializer())

    try:
        from .polars_serializer import (
            PolarsDataFrameSerializer,
            PolarsLazyFrameSerializer,
            PolarsSeriesSerializer,
        )
    except ImportError as e:
        if not library_is_absent(e, "polars"):
            raise
    else:
        registry.register(PolarsDataFrameSerializer())
        registry.register(PolarsSeriesSerializer())
        registry.register(PolarsLazyFrameSerializer())

    try:
        from .pandas_serializer import PandasDataFrameSerializer, PandasSeriesSerializer
    except ImportError as e:
        if not library_is_absent(e, "pandas"):
            raise
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

    Parameters
    ----------
    serializer : Serializer
        The serializer to add. It affects every store in the process that was not given
        its own registry.
    priority : int, optional
        The list index to insert at (``0`` is consulted first). When omitted, the
        serializer goes after the ones already registered and before the fallback.

    Raises
    ------
    ValueError
        If a serializer is already registered under the same ``type_tag``. To replace
        one, call ``default_registry.register(serializer, replace=True)``.

    See Also
    --------
    SerializerRegistry.copy : Add a serializer for one decoration only.
    """
    default_registry.register(serializer, priority=priority)
