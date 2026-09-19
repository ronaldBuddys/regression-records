"""Fallback serializer for arbitrary Python objects, via __getstate__/__dict__.

This is the lowest-priority serializer in the default registry: it is only reached for
objects no more specific serializer claimed.
"""

from __future__ import annotations

import importlib
from typing import Any

from .._object_state import object_state
from ..exceptions import IdentityResolutionError, SerializationError
from .base import DeserializationContext, SerializationContext


class GenericObjectSerializer:
    type_tag = "object"

    def can_handle(self, obj: Any) -> bool:
        # The fallback claims everything, so that an object with no recordable state
        # gets store()'s specific refusal rather than a generic "no serializer" error.
        return True

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        # Local import: avoids a circular import with the package's tree walker, which
        # imports this module (indirectly) while building the default registry.
        from . import to_json_tree

        cls = type(obj)
        if cls.__module__ == "__main__":
            raise SerializationError(
                f"cannot serialize instances of class '{cls.__qualname__}' defined in "
                "'__main__' - it cannot be re-imported from a fresh process later. "
                "Move the class into a regular importable module."
            )

        state = object_state(obj)
        if state is None:
            raise SerializationError(
                f"cannot serialize instances of '{cls.__qualname__}': it exposes no "
                "dict-shaped attribute state (a __slots__-only class, or a "
                "__getstate__() returning something other than a dict). Implement a "
                "custom Serializer for this type instead."
            )

        return {
            "class": {"module": cls.__module__, "qualname": cls.__qualname__},
            "state": to_json_tree(state, ctx),
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> Any:
        from . import from_json_tree

        class_info = blob["class"]
        try:
            obj = importlib.import_module(class_info["module"])
            for part in class_info["qualname"].split("."):
                obj = getattr(obj, part)
            cls = obj
        except (ImportError, AttributeError) as e:
            raise IdentityResolutionError(
                f"cannot import class '{class_info['qualname']}' from module "
                f"'{class_info['module']}': {e}"
            ) from e

        state = from_json_tree(blob["state"], ctx)
        instance = object.__new__(cls)
        if hasattr(instance, "__setstate__"):
            instance.__setstate__(state)
        else:
            instance.__dict__.update(state)
        return instance
