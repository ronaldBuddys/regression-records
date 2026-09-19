"""Fallback serializer for arbitrary Python objects, via __getstate__/__dict__.

This is the lowest-priority serializer in the default registry: it is only reached for
objects no more specific serializer claimed.

Loading is not safe on a store you do not trust. A record names the class to rebuild,
so loading it imports that module - running its top-level code - and hands the recorded
state to the class's ``__setstate__``. That is inherent in rebuilding objects by name,
and the reason to treat a store as you treat the code beside it: review changes to it.
"""

from __future__ import annotations

import importlib
from typing import Any

from .._object_state import object_state
from ..exceptions import IdentityResolutionError, SerializationError
from .base import DeserializationContext, SerializationContext


def class_reference(cls: type) -> dict[str, str]:
    """Return the ``{"module", "qualname"}`` a record uses to name ``cls``.

    Raises
    ------
    SerializationError
        If ``cls`` could not be imported by that name later: it is defined in
        ``__main__`` or inside a function. Refused now rather than at load time, when
        the record already exists and can never be read.
    """
    if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__:
        where = (
            "'__main__'" if cls.__module__ == "__main__" else "the body of a function"
        )
        raise SerializationError(
            f"cannot serialize instances of class '{cls.__qualname__}' defined in "
            f"{where} - it cannot be re-imported by name from a fresh process "
            "later. Move the class to the top level of a regular importable module."
        )
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def import_class(class_info: dict[str, str]) -> type:
    """Import the class a record names.

    Raises
    ------
    IdentityResolutionError
        If the module or attribute no longer exists, or the name resolves to something
        other than a class. The names come from a record file, so nothing but a class
        is ever handed back to be instantiated.
    """
    try:
        obj: Any = importlib.import_module(class_info["module"])
        for part in class_info["qualname"].split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as e:
        raise IdentityResolutionError(
            f"cannot import class '{class_info['qualname']}' from module "
            f"'{class_info['module']}': {e}"
        ) from e
    if not isinstance(obj, type):
        raise IdentityResolutionError(
            f"'{class_info['module']}.{class_info['qualname']}' is not a class"
        )
    return obj


class GenericObjectSerializer:
    """Stores an arbitrary object as its importable class plus its attribute state."""

    type_tag = "object"

    def can_handle(self, obj: Any) -> bool:
        """Claim every object, leaving :meth:`store` to refuse what it cannot record."""
        # The fallback claims everything, so that an object with no recordable state
        # gets store()'s specific refusal rather than a generic "no serializer" error.
        return True

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return the object's class location and its serialized attribute state.

        Raises ``SerializationError`` for classes defined in ``__main__``, which cannot
        be re-imported later, and for objects exposing no dict-shaped state.
        """
        # Local import: avoids a circular import with the package's tree walker, which
        # imports this module (indirectly) while building the default registry.
        from . import to_json_tree

        cls = type(obj)
        reference = class_reference(cls)

        state = object_state(obj)
        if state is None:
            raise SerializationError(
                f"cannot serialize instances of '{cls.__qualname__}': it exposes no "
                "dict-shaped attribute state (a __slots__-only class, or a "
                "__getstate__() returning something other than a dict). Implement a "
                "custom Serializer for this type instead."
            )

        return {
            "class": reference,
            "state": to_json_tree(state, ctx),
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> Any:
        """Re-import the class and restore an instance without calling ``__init__``.

        State goes through ``__setstate__`` when the instance has one, else into
        ``__dict__``. Raises ``IdentityResolutionError`` if the class cannot be
        imported.
        """
        from . import from_json_tree

        cls = import_class(blob["class"])

        state = from_json_tree(blob["state"], ctx)
        instance = object.__new__(cls)
        if hasattr(instance, "__setstate__"):
            instance.__setstate__(state)
        else:
            instance.__dict__.update(state)
        return instance
