"""Serializers for stdlib types that JSON cannot represent natively."""

from __future__ import annotations

import datetime
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
