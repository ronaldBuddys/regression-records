"""Serializers for stdlib types that JSON cannot represent natively.

Containers whose *shape* JSON can hold but whose *type* it cannot (tuple, set) live here
too, rather than being open-coded in the tree walker: a type tag with two owners - one
writing it, another reading it - is a tag that can quietly drift, and it also means a
third-party serializer claiming one of these tags is silently shadowed on load.
"""

from __future__ import annotations

import datetime
import json
import math
import zoneinfo
from typing import Any

from .._object_state import is_namedtuple
from ..exceptions import DeserializationError, IdentityResolutionError
from .base import DeserializationContext, SerializationContext


class DatetimeSerializer:
    """Stores ``datetime.datetime`` values as ISO 8601 strings.

    ISO 8601 carries a UTC offset, not a time zone, and has no ``fold``. So a
    ``ZoneInfo`` zone's key and a ``fold`` of 1 are stored beside it, as ``"tz"`` and
    ``"fold"``, only when present: any other datetime's payload is what it always was.
    Other ``tzinfo`` implementations come back as their fixed offset.
    """

    type_tag = "datetime.datetime"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``datetime.datetime`` instances."""
        return isinstance(obj, datetime.datetime)

    def store(self, obj: datetime.datetime, ctx: SerializationContext) -> dict:
        """Return the ISO 8601 string, plus the ``ZoneInfo`` key and ``fold`` if set."""
        payload: dict[str, Any] = {"value": obj.isoformat()}
        # A ZoneInfo built from a file rather than by name has no key to rebuild from.
        if isinstance(obj.tzinfo, zoneinfo.ZoneInfo) and obj.tzinfo.key is not None:
            payload["tz"] = obj.tzinfo.key
        if obj.fold:
            payload["fold"] = obj.fold
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> datetime.datetime:
        """Parse the stored string, restoring the ``ZoneInfo`` zone and ``fold``.

        Raises ``DeserializationError`` if the recorded zone is unknown on this machine.
        """
        value = datetime.datetime.fromisoformat(blob["value"])
        if "tz" in blob:
            # The string holds the local wall time, which with the zone and fold
            # names the same instant again - including an ambiguous one in a DST
            # transition, which the offset alone cannot tell the zone.
            try:
                zone = zoneinfo.ZoneInfo(blob["tz"])
            except zoneinfo.ZoneInfoNotFoundError as e:
                raise DeserializationError(
                    f"recorded time zone {blob['tz']!r} is not available here; "
                    "install tzdata or a system time zone database"
                ) from e
            value = value.replace(tzinfo=zone)
        return value.replace(fold=blob.get("fold", 0))


class DateSerializer:
    """Stores plain ``datetime.date`` values as ISO 8601 strings."""

    type_tag = "datetime.date"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``datetime.date`` instances that are not datetimes."""
        # datetime.datetime is a subclass of datetime.date - must be checked first.
        return isinstance(obj, datetime.date) and not isinstance(obj, datetime.datetime)

    def store(self, obj: datetime.date, ctx: SerializationContext) -> dict:
        """Return the date as an ISO 8601 ``YYYY-MM-DD`` string."""
        return {"value": obj.isoformat()}

    def load(self, blob: dict, ctx: DeserializationContext) -> datetime.date:
        """Parse the stored ISO 8601 string back into a date."""
        return datetime.date.fromisoformat(blob["value"])


class NonFiniteFloatSerializer:
    """Stores NaN and the infinities as tagged strings, keeping records strict JSON.

    JSON has no literal for them. Python's ``json`` writes ``NaN`` / ``Infinity``
    anyway, which ``jq``, JavaScript and most other parsers reject; a tag keeps the
    record readable by ordinary tools. Finite floats stay plain JSON numbers.
    """

    type_tag = "float"

    def can_handle(self, obj: Any) -> bool:
        """Claim Python floats that are NaN or infinite."""
        return isinstance(obj, float) and not math.isfinite(obj)

    def store(self, obj: float, ctx: SerializationContext) -> dict:
        """Return the value as ``"nan"``, ``"inf"`` or ``"-inf"``."""
        return {"value": repr(obj)}

    def load(self, blob: dict, ctx: DeserializationContext) -> float:
        """Parse the stored string back into a float."""
        return float(blob["value"])


class BytesSerializer:
    """Stores ``bytes`` values inline as hexadecimal strings."""

    type_tag = "bytes"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``bytes`` instances (not ``bytearray``)."""
        return isinstance(obj, bytes)

    def store(self, obj: bytes, ctx: SerializationContext) -> dict:
        """Return the bytes hex-encoded inline; no side file is written."""
        return {"hex": obj.hex()}

    def load(self, blob: dict, ctx: DeserializationContext) -> bytes:
        """Decode the stored hexadecimal string back into bytes."""
        return bytes.fromhex(blob["hex"])


class NamedTupleSerializer:
    """Stores named tuples as their class plus items, so the type survives.

    Registered ahead of :class:`TupleSerializer`. A named tuple stored as a plain tuple
    came back as one, and the comparison then reported a type mismatch against the
    freshly computed named tuple on every later run - a record that could be written
    but never validated.
    """

    type_tag = "namedtuple"

    def can_handle(self, obj: Any) -> bool:
        """Claim instances of named tuple classes."""
        return is_namedtuple(obj)

    def store(self, obj: tuple, ctx: SerializationContext) -> dict:
        """Return the class location and the serialized items, in field order.

        Raises ``SerializationError`` for a class that cannot be re-imported by name.
        """
        from . import to_json_tree
        from .object_serializer import class_reference

        return {
            "class": class_reference(type(obj)),
            "items": [to_json_tree(v, ctx) for v in obj],
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> tuple:
        """Re-import the class and rebuild the named tuple from the stored items.

        Raises ``IdentityResolutionError`` if the class cannot be imported or is not a
        named tuple class.
        """
        from . import from_json_tree
        from .object_serializer import import_class

        cls = import_class(blob["class"])
        if not (issubclass(cls, tuple) and hasattr(cls, "_fields")):
            raise IdentityResolutionError(
                f"'{cls.__module__}.{cls.__qualname__}' is not a named tuple class"
            )
        # tuple.__new__ rather than cls(*items): the class is named by a record file,
        # and calling it would run whatever constructor that name resolves to.
        return tuple.__new__(cls, [from_json_tree(v, ctx) for v in blob["items"]])


class TupleSerializer:
    """Preserves tuple-ness, which JSON arrays cannot.

    A tuple is not interchangeable with a list: a function that started returning one
    where it used to return the other has changed its contract, and the comparison layer
    reports that as a type mismatch.
    """

    type_tag = "tuple"

    def can_handle(self, obj: Any) -> bool:
        """Claim tuples; named tuples never get here, other subclasses do."""
        return isinstance(obj, tuple)

    def store(self, obj: tuple, ctx: SerializationContext) -> dict:
        """Return the items, each serialized through the tree walker, in order."""
        from . import to_json_tree

        return {"items": [to_json_tree(v, ctx) for v in obj]}

    def load(self, blob: dict, ctx: DeserializationContext) -> tuple:
        """Rebuild a plain tuple from the stored items.

        A tuple subclass other than a named tuple comes back as a plain tuple, which
        is also how the comparison treats it.
        """
        from . import from_json_tree

        return tuple(from_json_tree(v, ctx) for v in blob["items"])


class SetSerializer:
    """Stores ``set`` values as a deterministically sorted list of their items."""

    type_tag = "set"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``set`` instances (not ``frozenset``)."""
        return isinstance(obj, set)

    def store(self, obj: set, ctx: SerializationContext) -> dict:
        """Return the serialized items, sorted by their canonical JSON text."""
        from . import to_json_tree

        items = [to_json_tree(v, ctx) for v in obj]
        # Set iteration order isn't stable across processes; sort deterministically so
        # identical sets always serialize - and so hash - identically.
        items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return {"items": items}

    def load(self, blob: dict, ctx: DeserializationContext) -> set:
        """Rebuild the set from the stored items."""
        from . import from_json_tree

        return {from_json_tree(v, ctx) for v in blob["items"]}
