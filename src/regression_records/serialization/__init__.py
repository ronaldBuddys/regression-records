"""Convert arbitrary Python objects to and from a JSON-safe tree.

Leaf values JSON cannot hold (arrays, dataframes, datetimes, ...) are handed to a
:class:`~regression_records.serialization.base.SerializerRegistry`, which may offload them
to external files. Any other object is stored as its class plus its attributes, by the
object path in :mod:`~regression_records.serialization.object_serializer` - no
serializer is needed for a user's own class.

The public entry points are :func:`to_json_tree` / :func:`from_json_tree`, plus
:func:`register_serializer` for extending the module-level :data:`default_registry`
with support for additional types.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .._attributes import stored_by_attributes
from .._optional import library_is_absent
from ..exceptions import SerializationError
from .base import (
    OBJECT_TAG,
    RESERVED_TAGS,
    DeserializationContext,
    SerializationContext,
    Serializer,
    SerializerRegistry,
)
from .object_serializer import load_object, store_object

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
    return {key: to_json_tree(value, ctx) for key, value in obj.items()}


def _list_to_json_tree(obj: list, ctx: SerializationContext) -> list:
    return [to_json_tree(v, ctx) for v in obj]


def _delegate(obj: Any, ctx: SerializationContext) -> dict:
    """Store ``obj`` through the leaf serializer that claims it, else by its attributes.

    The registry is asked first, so a leaf serializer - built in or registered - wins
    over the object path for any type it claims.
    """
    serializer = ctx.registry.find(obj)
    if serializer is None:
        return _store_by_attributes(obj, ctx)
    payload = serializer.store(obj, ctx)
    if TYPE_KEY in payload:
        # The tag decides which serializer reads the value back, so a payload must not
        # be able to replace it - the same rule a recorded dict is held to.
        raise SerializationError(
            f"{type(serializer).__name__}.store() returned the reserved key "
            f"{TYPE_KEY!r}; a serializer names its type through type_tag"
        )
    return {TYPE_KEY: serializer.type_tag, **payload}


def _store_by_attributes(obj: Any, ctx: SerializationContext) -> dict:
    """Store ``obj`` as its class plus its attributes, or refuse it.

    Raises ``SerializationError`` for a value that is neither a leaf nor rebuildable
    from attributes: a C type (a lock, a file handle, a type from a compiled
    extension), a function, a class or a module.
    """
    if not stored_by_attributes(obj):
        cls = type(obj)
        raise SerializationError(
            f"cannot record a value of type '{cls.__module__}.{cls.__qualname__}': it is "
            "not a supported leaf type, and it cannot be rebuilt from its attributes "
            "(functions, classes, modules, and C types such as locks, file handles or "
            "compiled-extension objects). Record the data it holds instead, or "
            "register a leaf Serializer for the type."
        )
    return {TYPE_KEY: OBJECT_TAG, **store_object(obj, ctx)}


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
            "otherwise record a value without it."
        )
    ctx.active.add(key)
    try:
        return convert(obj, ctx)
    finally:
        ctx.active.discard(key)


def to_json_tree(obj: Any, ctx: SerializationContext) -> Any:
    """Recursively convert ``obj`` into a JSON-safe structure.

    Only the values JSON represents faithfully - primitives, lists, and string-keyed
    dicts - are handled here. Leaf values go to ``ctx.registry`` (writing external files
    under ``ctx.data_dir`` as needed), and any other object is stored as its class plus
    its attributes, each converted by these same rules.

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
        If a value is neither a leaf nor rebuildable from its attributes; a dict uses
        the reserved key ``"__type__"``; a serializer's payload uses that key; a
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
        # A string-keyed dict is a JSON object as it stands. Any other key needs the
        # tagged pair list of the dict serializer, which keeps key types.
        if all(isinstance(key, str) for key in obj):
            return _descend(obj, ctx, _dict_to_json_tree)
        return _descend(obj, ctx, _delegate)

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
            # The tag chose the serializer; its load() gets back exactly what its
            # store() returned. Objects have no serializer: the tag is the object path.
            tag = tree[TYPE_KEY]
            payload = {k: v for k, v in tree.items() if k != TYPE_KEY}
            if tag in RESERVED_TAGS:
                return load_object(tag, payload, ctx)
            return ctx.registry.by_tag(tag).load(payload, ctx)
        return {k: from_json_tree(v, ctx) for k, v in tree.items()}
    if isinstance(tree, list):
        return [from_json_tree(v, ctx) for v in tree]
    return tree


def _build_default_registry() -> SerializerRegistry:
    registry = SerializerRegistry()

    from .numpy_serializer import NumpyArraySerializer, NumpyScalarSerializer
    from .primitives import (
        BytearraySerializer,
        BytesSerializer,
        ComplexSerializer,
        DateSerializer,
        DatetimeSerializer,
        DecimalSerializer,
        DictSerializer,
        EnumSerializer,
        FrozenSetSerializer,
        MappingProxySerializer,
        NonFiniteFloatSerializer,
        PathSerializer,
        SetSerializer,
        TimedeltaSerializer,
        TimeSerializer,
        TupleSerializer,
    )

    registry.register(NumpyScalarSerializer())
    registry.register(NumpyArraySerializer())
    registry.register(DatetimeSerializer())
    registry.register(DateSerializer())
    registry.register(NonFiniteFloatSerializer())
    registry.register(BytesSerializer())
    registry.register(TupleSerializer())
    registry.register(SetSerializer())
    registry.register(FrozenSetSerializer())
    registry.register(BytearraySerializer())
    registry.register(TimeSerializer())
    registry.register(TimedeltaSerializer())
    registry.register(DecimalSerializer())
    registry.register(PathSerializer())
    registry.register(ComplexSerializer())
    registry.register(EnumSerializer())
    registry.register(DictSerializer())
    registry.register(MappingProxySerializer())

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
        from .pandas_serializer import (
            PandasDataFrameSerializer,
            PandasSeriesSerializer,
            PandasTimedeltaSerializer,
            PandasTimestampSerializer,
        )
    except ImportError as e:
        if not library_is_absent(e, "pandas"):
            raise
    else:
        registry.register(PandasDataFrameSerializer())
        registry.register(PandasSeriesSerializer())
        # First: they subclass datetime / timedelta, whose serializers would otherwise
        # claim them and drop both the nanoseconds and the type.
        registry.register(PandasTimestampSerializer(), priority=0)
        registry.register(PandasTimedeltaSerializer(), priority=0)

    return registry


default_registry = _build_default_registry()


def register_serializer(serializer: Serializer, *, priority: int | None = None) -> None:
    """Register ``serializer`` on the module-level :data:`default_registry`.

    A serializer is for a *leaf* type: a value that is not an object to store by its
    attributes, such as a C type from a compiled extension. A user's own class needs
    none - it is stored by its attributes. The serializer goes after the built-in ones,
    and is consulted before the object path, so it also wins for any class it claims;
    pass an explicit ``priority`` index to go ahead of the built-ins as well.

    Parameters
    ----------
    serializer : Serializer
        The serializer to add. It affects every store in the process that was not given
        its own registry.
    priority : int, optional
        The list index to insert at (``0`` is consulted first). When omitted, the
        serializer goes after the ones already registered.

    Raises
    ------
    ValueError
        If a serializer is already registered under the same ``type_tag``, or the tag
        is one the object path reserves. To replace one, call
        ``default_registry.register(serializer, replace=True)``.

    See Also
    --------
    SerializerRegistry.copy : Add a serializer for one decoration only.
    """
    default_registry.register(serializer, priority=priority)
