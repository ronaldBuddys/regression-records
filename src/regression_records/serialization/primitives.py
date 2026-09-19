"""Serializers for stdlib types that JSON cannot represent natively.

Containers whose *shape* JSON can hold but whose *type* it cannot (tuple, set) live here
too, rather than being open-coded in the tree walker: a type tag with two owners - one
writing it, another reading it - is a tag that can quietly drift, and it also means a
third-party serializer claiming one of these tags is silently shadowed on load.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from .base import DeserializationContext, SerializationContext


class DatetimeSerializer:
    type_tag = "datetime.datetime"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, datetime.datetime)

    def store(self, obj: datetime.datetime, ctx: SerializationContext) -> dict:
        return {"value": obj.isoformat()}

    def load(self, blob: dict, ctx: DeserializationContext) -> datetime.datetime:
        return datetime.datetime.fromisoformat(blob["value"])


class DateSerializer:
    type_tag = "datetime.date"

    def can_handle(self, obj: Any) -> bool:
        # datetime.datetime is a subclass of datetime.date - must be checked first.
        return isinstance(obj, datetime.date) and not isinstance(obj, datetime.datetime)

    def store(self, obj: datetime.date, ctx: SerializationContext) -> dict:
        return {"value": obj.isoformat()}

    def load(self, blob: dict, ctx: DeserializationContext) -> datetime.date:
        return datetime.date.fromisoformat(blob["value"])


class BytesSerializer:
    type_tag = "bytes"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, bytes)

    def store(self, obj: bytes, ctx: SerializationContext) -> dict:
        return {"hex": obj.hex()}

    def load(self, blob: dict, ctx: DeserializationContext) -> bytes:
        return bytes.fromhex(blob["hex"])


class TupleSerializer:
    """Preserves tuple-ness, which JSON arrays cannot.

    A tuple is not interchangeable with a list: a function that started returning one
    where it used to return the other has changed its contract, and the comparison layer
    reports that as a type mismatch.
    """

    type_tag = "tuple"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, tuple)

    def store(self, obj: tuple, ctx: SerializationContext) -> dict:
        from . import to_json_tree

        return {"items": [to_json_tree(v, ctx) for v in obj]}

    def load(self, blob: dict, ctx: DeserializationContext) -> tuple:
        from . import from_json_tree

        return tuple(from_json_tree(v, ctx) for v in blob["items"])


class SetSerializer:
    type_tag = "set"

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, set)

    def store(self, obj: set, ctx: SerializationContext) -> dict:
        from . import to_json_tree

        items = [to_json_tree(v, ctx) for v in obj]
        # Set iteration order isn't stable across processes; sort deterministically so
        # identical sets always serialize - and so hash - identically.
        items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return {"items": items}

    def load(self, blob: dict, ctx: DeserializationContext) -> set:
        from . import from_json_tree

        return {from_json_tree(v, ctx) for v in blob["items"]}
