"""The RegressionRecord data model: a persistent representation of one invocation."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._version import __version__
from .identity import FunctionIdentity
from .serialization import DeserializationContext, from_json_tree

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SchemaInfo:
    """Version metadata stored on every record, enabling future format migration."""

    schema_version: int = SCHEMA_VERSION
    library_version: str = __version__


@dataclass(frozen=True)
class SerializedValue:
    """A value already converted to the JSON-safe tree produced by ``to_json_tree``."""

    data: Any

    def load(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the original Python object."""
        return from_json_tree(self.data, ctx)


@dataclass(frozen=True)
class ExceptionRecord:
    """A captured exception.

    Only ``exc_type`` participates in regression comparison by default - messages
    frequently carry nondeterministic detail (addresses, paths, timings) - and
    ``traceback_text`` is retained purely for display.
    """

    exc_type: str
    message: str
    traceback_text: str


@dataclass(frozen=True)
class Invocation:
    """The serialized positional and keyword arguments of one recorded call."""

    args: SerializedValue
    kwargs: SerializedValue


@dataclass
class RegressionRecord:
    """A persistent representation of one recorded function/method invocation.

    Exactly one of ``output``/``exception`` is set, per whether the call succeeded
    (see :attr:`succeeded`). ``state_before``/``state_after`` are only set for
    recorded methods.
    """

    schema: SchemaInfo
    record_id: str
    input_hash: str
    created_at: str
    identity: FunctionIdentity
    invocation: Invocation
    output: SerializedValue | None = None
    exception: ExceptionRecord | None = None
    state_before: SerializedValue | None = None
    state_after: SerializedValue | None = None
    tags: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Return whether the recorded call returned rather than raised."""
        return self.exception is None

    @classmethod
    def new(
        cls,
        *,
        identity: FunctionIdentity,
        input_hash: str,
        invocation: Invocation,
        output: SerializedValue | None = None,
        exception: ExceptionRecord | None = None,
        state_before: SerializedValue | None = None,
        state_after: SerializedValue | None = None,
        tags: dict | None = None,
        record_id: str | None = None,
    ) -> RegressionRecord:
        """Build a record stamped with the current schema, UTC time and a fresh id.

        ``record_id`` defaults to a random UUID4 hex; pass one to keep an existing id.
        """
        return cls(
            schema=SchemaInfo(),
            record_id=record_id or uuid.uuid4().hex,
            input_hash=input_hash,
            created_at=datetime.now(UTC).isoformat(),
            identity=identity,
            invocation=invocation,
            output=output,
            exception=exception,
            state_before=state_before,
            state_after=state_after,
            tags=tags or {},
        )

    def load_output(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the recorded return value.

        Raises ``ValueError`` when the recorded call raised instead of returning.
        """
        if self.output is None:
            raise ValueError("this record has no output (the call raised an exception)")
        return self.output.load(ctx)

    def load_args(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the recorded positional arguments."""
        return self.invocation.args.load(ctx)

    def load_kwargs(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the recorded keyword arguments."""
        return self.invocation.kwargs.load(ctx)

    def load_state_before(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the instance state captured before the call.

        Raises ``ValueError`` when the record is not a method recording.
        """
        if self.state_before is None:
            raise ValueError("this record has no state_before (not a method recording)")
        return self.state_before.load(ctx)

    def load_state_after(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the instance state captured after the call.

        Raises ``ValueError`` when the record is not a method recording.
        """
        if self.state_after is None:
            raise ValueError("this record has no state_after (not a method recording)")
        return self.state_after.load(ctx)

    def to_dict(self) -> dict:
        """Convert to a plain JSON-safe dict (the on-disk record format).

        Optional fields are only present as keys when set, so a ``None`` Python
        return value (``output.data is None``) can be told apart from "no output was
        recorded" (``"output" not in dict``, e.g. because the call raised).
        """
        out: dict[str, Any] = {
            # asdict() on the flat metadata dataclasses, so adding a field to any of
            # them can't leave the two directions of the round trip disagreeing.
            "schema": asdict(self.schema),
            "record_id": self.record_id,
            "input_hash": self.input_hash,
            "created_at": self.created_at,
            "identity": asdict(self.identity),
            "invocation": {
                "args": self.invocation.args.data,
                "kwargs": self.invocation.kwargs.data,
            },
            "tags": self.tags,
        }
        if self.output is not None:
            out["output"] = self.output.data
        if self.exception is not None:
            out["exception"] = asdict(self.exception)
        if self.state_before is not None:
            out["state_before"] = self.state_before.data
        if self.state_after is not None:
            out["state_after"] = self.state_after.data
        return out

    @classmethod
    def from_dict(cls, d: dict) -> RegressionRecord:
        """Rebuild a record from the dict produced by :meth:`to_dict`.

        Optional fields absent from ``d`` come back as ``None``.
        """
        identity = FunctionIdentity(**d["identity"])
        invocation = Invocation(
            args=SerializedValue(d["invocation"]["args"]),
            kwargs=SerializedValue(d["invocation"]["kwargs"]),
        )
        return cls(
            schema=SchemaInfo(**d["schema"]),
            record_id=d["record_id"],
            input_hash=d["input_hash"],
            created_at=d["created_at"],
            identity=identity,
            invocation=invocation,
            output=SerializedValue(d["output"]) if "output" in d else None,
            exception=ExceptionRecord(**d["exception"]) if "exception" in d else None,
            state_before=SerializedValue(d["state_before"])
            if "state_before" in d
            else None,
            state_after=SerializedValue(d["state_after"])
            if "state_after" in d
            else None,
            tags=d.get("tags", {}),
        )
