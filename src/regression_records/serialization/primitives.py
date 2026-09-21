"""Serializers for stdlib types that JSON cannot represent natively.

Containers whose *shape* JSON can hold but whose *type* it cannot (tuple, set) live here
too, rather than being open-coded in the tree walker: a type tag with two owners - one
writing it, another reading it - is a tag that can quietly drift, and it also means a
third-party serializer claiming one of these tags is silently shadowed on load.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import json
import math
import pathlib
import zoneinfo
from typing import Any

from .._object_state import is_namedtuple
from ..exceptions import (
    DeserializationError,
    IdentityResolutionError,
    SerializationError,
)
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


class FrozenSetSerializer(SetSerializer):
    """Stores ``frozenset`` values as ``set`` values are, keeping the type."""

    type_tag = "frozenset"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``frozenset`` instances."""
        return isinstance(obj, frozenset)

    def load(self, blob: dict, ctx: DeserializationContext) -> frozenset:
        """Rebuild the frozenset from the stored items."""
        return frozenset(super().load(blob, ctx))


class BytearraySerializer:
    """Stores ``bytearray`` values inline as hexadecimal strings, keeping the type."""

    type_tag = "bytearray"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``bytearray`` instances."""
        return isinstance(obj, bytearray)

    def store(self, obj: bytearray, ctx: SerializationContext) -> dict:
        """Return the bytes hex-encoded inline."""
        return {"hex": obj.hex()}

    def load(self, blob: dict, ctx: DeserializationContext) -> bytearray:
        """Decode the stored hexadecimal string back into a bytearray."""
        return bytearray.fromhex(blob["hex"])


class TimeSerializer:
    """Stores ``datetime.time`` values as ISO 8601 strings, plus ``fold`` if set."""

    type_tag = "datetime.time"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``datetime.time`` instances."""
        return isinstance(obj, datetime.time)

    def store(self, obj: datetime.time, ctx: SerializationContext) -> dict:
        """Return the time as an ISO 8601 string, UTC offset included if aware."""
        payload: dict[str, Any] = {"value": obj.isoformat()}
        if obj.fold:
            payload["fold"] = obj.fold
        return payload

    def load(self, blob: dict, ctx: DeserializationContext) -> datetime.time:
        """Parse the stored ISO 8601 string back into a time."""
        value = datetime.time.fromisoformat(blob["value"])
        return value.replace(fold=blob.get("fold", 0))


class TimedeltaSerializer:
    """Stores ``datetime.timedelta`` values as their exact integer components."""

    type_tag = "datetime.timedelta"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``datetime.timedelta`` instances."""
        return isinstance(obj, datetime.timedelta)

    def store(self, obj: datetime.timedelta, ctx: SerializationContext) -> dict:
        """Return days, seconds and microseconds, as the timedelta normalizes them."""
        return {
            "days": obj.days,
            "seconds": obj.seconds,
            "microseconds": obj.microseconds,
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> datetime.timedelta:
        """Rebuild the timedelta from its components."""
        return datetime.timedelta(
            days=blob["days"],
            seconds=blob["seconds"],
            microseconds=blob["microseconds"],
        )


class DecimalSerializer:
    """Stores ``decimal.Decimal`` values as strings, so no digit or exponent is lost.

    ``str`` keeps the coefficient and exponent exactly - ``Decimal("1.10")`` stays
    distinct from ``Decimal("1.1")`` - as well as signed zeros, NaN and infinities.
    """

    type_tag = "decimal.Decimal"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``decimal.Decimal`` instances."""
        return isinstance(obj, decimal.Decimal)

    def store(self, obj: decimal.Decimal, ctx: SerializationContext) -> dict:
        """Return the decimal's string form."""
        return {"value": str(obj)}

    def load(self, blob: dict, ctx: DeserializationContext) -> decimal.Decimal:
        """Parse the stored string back into a decimal."""
        return decimal.Decimal(blob["value"])


#: The pathlib classes a record may name. The concrete ones load as ``Path`` - the
#: concrete class for the loading platform - so a record stays loadable elsewhere.
_PATH_CLASSES: dict[str, type[pathlib.PurePath]] = {
    "Path": pathlib.Path,
    "PosixPath": pathlib.Path,
    "WindowsPath": pathlib.Path,
    "PurePath": pathlib.PurePath,
    "PurePosixPath": pathlib.PurePosixPath,
    "PureWindowsPath": pathlib.PureWindowsPath,
}
_PATHLIB_TYPES = {
    pathlib.Path,
    pathlib.PosixPath,
    pathlib.WindowsPath,
    pathlib.PurePath,
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
}


class PathSerializer:
    """Stores ``pathlib`` paths as their string and class name.

    Only the ``pathlib`` classes themselves are claimed; a subclass goes on to the
    generic object serializer.
    """

    type_tag = "pathlib.Path"

    def can_handle(self, obj: Any) -> bool:
        """Claim instances of the ``pathlib`` path classes."""
        return type(obj) in _PATHLIB_TYPES

    def store(self, obj: pathlib.PurePath, ctx: SerializationContext) -> dict:
        """Return the path as a string, with its class name."""
        return {"value": str(obj), "class": type(obj).__name__}

    def load(self, blob: dict, ctx: DeserializationContext) -> pathlib.PurePath:
        """Rebuild the path; a concrete one as ``Path`` for this platform.

        Raises ``DeserializationError`` for a class name that is not a pathlib class.
        """
        cls = _PATH_CLASSES.get(blob["class"])
        if cls is None:
            raise DeserializationError(f"not a pathlib class: {blob['class']!r}")
        return cls(blob["value"])


class ComplexSerializer:
    """Stores Python ``complex`` values as their real and imaginary parts."""

    type_tag = "complex"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``complex`` instances."""
        return isinstance(obj, complex)

    def store(self, obj: complex, ctx: SerializationContext) -> dict:
        """Return both parts, through the tree walker so NaN and infinity are tagged."""
        from . import to_json_tree

        return {
            "real": to_json_tree(obj.real, ctx),
            "imag": to_json_tree(obj.imag, ctx),
        }

    def load(self, blob: dict, ctx: DeserializationContext) -> complex:
        """Rebuild the complex number from its parts."""
        from . import from_json_tree

        return complex(
            from_json_tree(blob["real"], ctx), from_json_tree(blob["imag"], ctx)
        )


class EnumSerializer:
    """Stores ``Enum`` members as their class plus member name.

    ``IntEnum`` and ``StrEnum`` members never get here: they are recorded as the
    ``int`` / ``str`` they are. A composite ``Flag`` value, which has no single member
    name, is refused.
    """

    type_tag = "enum"

    def can_handle(self, obj: Any) -> bool:
        """Claim ``enum.Enum`` members."""
        return isinstance(obj, enum.Enum)

    def store(self, obj: enum.Enum, ctx: SerializationContext) -> dict:
        """Return the class location and the member's name.

        Raises ``SerializationError`` for a class that cannot be re-imported by name,
        or a value that is not a single named member.
        """
        from .object_serializer import class_reference

        cls = type(obj)
        if cls.__members__.get(obj.name) is not obj:
            raise SerializationError(
                f"{cls.__qualname__} value {obj!r} is not a single named member (a "
                "combination of Flag members, for instance), so it cannot be recorded "
                "by name; record its .value instead"
            )
        return {"class": class_reference(cls), "member": obj.name}

    def load(self, blob: dict, ctx: DeserializationContext) -> enum.Enum:
        """Re-import the class and look the member up by name.

        Raises ``IdentityResolutionError`` if the class cannot be imported, is not an
        ``Enum``, or no longer has that member.
        """
        from .object_serializer import import_class

        cls = import_class(blob["class"])
        if not issubclass(cls, enum.Enum):
            raise IdentityResolutionError(
                f"'{cls.__module__}.{cls.__qualname__}' is not an Enum class"
            )
        try:
            return cls[blob["member"]]
        except KeyError:
            raise IdentityResolutionError(
                f"'{cls.__module__}.{cls.__qualname__}' has no member "
                f"{blob['member']!r}"
            ) from None
