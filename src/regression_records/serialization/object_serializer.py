"""The object path: any instance stored as its importable class plus its attributes.

:data:`OBJECT_PATH` is not in any registry's search order. The tree walker falls back to
it for every value no leaf serializer claims, and every registry holds it under the tags
it reads, so a user's own class never needs a serializer of its own.

Loading is not safe on a store you do not trust. A record names the class to rebuild, so
loading it imports that module - running its top-level code - and sets the recorded
attributes on a new instance. That is inherent in rebuilding objects by name, and the
reason to treat a store as you treat the code beside it: review changes to it.
"""

from __future__ import annotations

from typing import Any

from .._attributes import (
    attributes,
    class_name,
    class_reference,
    import_class,
    is_namedtuple_class,
    require_stored_by_attributes,
    restore,
    stored_by_attributes,
)
from ..exceptions import DeserializationError
from .base import OBJECT_TAG, DeserializationContext, SerializationContext


class ObjectSerializer:
    """Stores any object as its class location plus its serialized attributes."""

    type_tag = OBJECT_TAG

    def can_handle(self, obj: Any) -> bool:
        """Claim values rebuildable from their attributes (never asked by a registry)."""
        return stored_by_attributes(obj)

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return ``obj``'s class location and its serialized attributes.

        Raises
        ------
        SerializationError
            For a value that cannot be rebuilt from its attributes, a class defined in
            ``__main__`` or inside a function, or an attribute that cannot itself be
            recorded.
        """
        # Local import: the tree walker imports this module.
        from . import to_json_tree

        require_stored_by_attributes(obj, "record a value")
        reference = class_reference(type(obj))
        return {"class": reference, "attrs": to_json_tree(attributes(obj), ctx)}

    def load(self, blob: dict, ctx: DeserializationContext) -> Any:
        """Re-import the class a payload names and rebuild the instance.

        The payload says which format wrote it: ``"attrs"`` is the current one; a
        schema version 2 object holds ``"state"`` - its ``__getstate__()`` then - set
        back the same way, and a schema version 2 named tuple (tagged ``"namedtuple"``)
        holds its fields by position as ``"items"``.

        Raises
        ------
        IdentityResolutionError
            If the class cannot be imported.
        DeserializationError
            If the recorded attributes no longer fit the class.
        """
        from . import from_json_tree

        cls = import_class(blob["class"])
        if "items" in blob:
            return restore(cls, _legacy_fields(cls, from_json_tree(blob["items"], ctx)))
        tree = blob["attrs"] if "attrs" in blob else blob["state"]
        return restore(cls, from_json_tree(tree, ctx))


def _legacy_fields(cls: type, items: list) -> dict[str, Any]:
    """Name a schema version 2 named tuple's positional items by the class's fields."""
    if not is_namedtuple_class(cls):
        raise DeserializationError(f"'{class_name(cls)}' is not a named tuple class")
    if len(items) != len(cls._fields):
        raise DeserializationError(
            f"'{class_name(cls)}' has {len(cls._fields)} fields, the record holds "
            f"{len(items)}; re-record it"
        )
    return dict(zip(cls._fields, items, strict=True))


#: The one object serializer: what the tree walker falls back to, and what every
#: registry loads the reserved tags with.
OBJECT_PATH = ObjectSerializer()
