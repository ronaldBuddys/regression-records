"""The object path: any instance stored as its importable class plus its attributes.

This is not a registered serializer. The tree walker takes it for every value that no
leaf serializer claims and that :func:`~regression_records._attributes.stored_by_attributes`
accepts, so a user's own class never needs a serializer of its own.

Loading is not safe on a store you do not trust. A record names the class to rebuild, so
loading it imports that module - running its top-level code - and sets the recorded
attributes on a new instance. That is inherent in rebuilding objects by name, and the
reason to treat a store as you treat the code beside it: review changes to it.
"""

from __future__ import annotations

from typing import Any

from .._attributes import attributes, class_reference, import_class, restore
from ..exceptions import DeserializationError
from .base import (
    LEGACY_NAMEDTUPLE_TAG,
    DeserializationContext,
    SerializationContext,
)


def store_object(obj: Any, ctx: SerializationContext) -> dict:
    """Return ``obj``'s class location and its serialized attributes, untagged.

    Raises
    ------
    SerializationError
        For a class defined in ``__main__`` or inside a function, which cannot be
        re-imported later, or an attribute that cannot itself be recorded.
    """
    # Local import: the tree walker imports this module.
    from . import to_json_tree

    reference = class_reference(type(obj))
    return {"class": reference, "attrs": to_json_tree(attributes(obj), ctx)}


def load_object(tag: str, payload: dict, ctx: DeserializationContext) -> Any:
    """Re-import the class a payload names and rebuild the instance from its attributes.

    Parameters
    ----------
    tag : str
        :data:`OBJECT_TAG`, or :data:`LEGACY_NAMEDTUPLE_TAG` for a schema version 2
        named tuple, whose items were stored by position.
    payload : dict
        What :func:`store_object` returned. A schema version 2 object holds ``"state"``
        - its ``__getstate__()`` then - in place of ``"attrs"``, and is set back the
        same way.
    ctx : DeserializationContext
        Resolves the side files nested attributes reference.

    Returns
    -------
    Any
        The rebuilt instance; ``__init__`` is not called.

    Raises
    ------
    IdentityResolutionError
        If the class cannot be imported.
    DeserializationError
        If the recorded attributes no longer fit the class.
    """
    from . import from_json_tree

    cls = import_class(payload["class"])
    if tag == LEGACY_NAMEDTUPLE_TAG:
        fields = getattr(cls, "_fields", None)
        if not (issubclass(cls, tuple) and fields is not None):
            raise DeserializationError(
                f"'{cls.__module__}.{cls.__qualname__}' is not a named tuple class"
            )
        items = [from_json_tree(v, ctx) for v in payload["items"]]
        if len(items) != len(fields):
            raise DeserializationError(
                f"'{cls.__module__}.{cls.__qualname__}' has {len(fields)} fields, the "
                f"record holds {len(items)}; re-record it"
            )
        return restore(cls, dict(zip(fields, items, strict=True)))
    tree = payload["attrs"] if "attrs" in payload else payload["state"]
    return restore(cls, from_json_tree(tree, ctx))
