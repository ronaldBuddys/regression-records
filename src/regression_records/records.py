"""The RegressionRecord data model: a persistent representation of one invocation."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._version import __version__
from .exceptions import CorruptRecordError
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
    """A value already converted to the JSON-safe tree produced by ``to_json_tree``.

    Attributes
    ----------
    data : Any
        The JSON-safe tree. Side files it references are not held here.
    """

    data: Any

    def load(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the original Python object.

        Parameters
        ----------
        ctx : DeserializationContext
            Resolves the tree's file references; see
            :meth:`RecordStore.deserialization_context
            <regression_records.store.RecordStore.deserialization_context>`.

        Returns
        -------
        Any
            The value rebuilt by
            :func:`~regression_records.serialization.from_json_tree`.

        Raises
        ------
        DeserializationError
            If the tree names an unregistered type tag, or a serializer cannot load its
            value (:class:`~regression_records.exceptions.CorruptRecordError` for a
            missing or out-of-store side file).
        IdentityResolutionError
            If a recorded object's class can no longer be imported by name.
        """
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

    Attributes
    ----------
    schema : SchemaInfo
        The record format version and the library version that wrote it.
    record_id : str
        A unique id for this recording (a UUID4 hex from :meth:`new`), telling apart
        successive recordings of the same inputs.
    input_hash : str
        The sha256 hex digest of the serialized arguments - the key calls are matched
        on. See :func:`~regression_records.matching.compute_input_hash`.
    created_at : str
        When the record was built, as an ISO 8601 timestamp in UTC.
    identity : FunctionIdentity
        The function or method that was called.
    invocation : Invocation
        The serialized positional and keyword arguments, receiver excluded.
    output : SerializedValue or None, default None
        The serialized return value; None when the call raised. A call that *returned*
        ``None`` has a ``SerializedValue`` whose ``data`` is None.
    exception : ExceptionRecord or None, default None
        The exception the call raised; None when it returned.
    state_before : SerializedValue or None, default None
        The receiver's serialized state before the call (methods only).
    state_after : SerializedValue or None, default None
        The receiver's serialized state after the call (methods only).
    tags : dict, default {}
        Free-form metadata from the decoration's ``tags=``.
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
        """Return whether the recorded call returned rather than raised.

        Returns
        -------
        bool
            True when ``exception`` is None.
        """
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

        Parameters
        ----------
        identity : FunctionIdentity
            The function or method that was called.
        input_hash : str
            The hash of the serialized arguments, from
            :func:`~regression_records.matching.compute_input_hash`.
        invocation : Invocation
            The serialized arguments themselves.
        output : SerializedValue, optional
            The serialized return value, for a call that returned.
        exception : ExceptionRecord, optional
            The captured exception, for a call that raised. That exactly one of
            ``output`` and ``exception`` is given is the caller's responsibility; it
            is not checked.
        state_before, state_after : SerializedValue, optional
            The receiver's serialized state around the call, for a method.
        tags : dict, optional
            Free-form metadata; an empty dict when omitted.
        record_id : str, optional
            An id to reuse - replacing a record in place keeps its filename this way.

        Returns
        -------
        RegressionRecord
            With ``schema`` set to the current :class:`SchemaInfo` and ``created_at``
            to the current UTC time in ISO 8601 form.
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

        Parameters
        ----------
        ctx : DeserializationContext
            From :meth:`RecordStore.deserialization_context
            <regression_records.store.RecordStore.deserialization_context>` for this
            record's identity.

        Returns
        -------
        Any
            The value the recorded call returned.

        Raises
        ------
        ValueError
            If the record has no output.
        DeserializationError, IdentityResolutionError
            As for :meth:`SerializedValue.load`.
        """
        if self.output is None:
            raise ValueError("this record has no output (the call raised an exception)")
        return self.output.load(ctx)

    def load_args(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the recorded positional arguments.

        Parameters
        ----------
        ctx : DeserializationContext
            As for :meth:`load_output`.

        Returns
        -------
        list
            The arguments as bound against the signature, defaults applied - so one
            passed by keyword may appear here. Empty for a method recorded by the
            decorators: with the receiver left out, its arguments are all recorded by
            name.

        Raises
        ------
        DeserializationError, IdentityResolutionError
            As for :meth:`SerializedValue.load`.
        """
        return self.invocation.args.load(ctx)

    def load_kwargs(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the recorded keyword arguments.

        Parameters
        ----------
        ctx : DeserializationContext
            As for :meth:`load_output`.

        Returns
        -------
        dict
            The keyword-only and ``**kwargs`` arguments of a function; every argument
            but the receiver, by parameter name, for a method.

        Raises
        ------
        DeserializationError, IdentityResolutionError
            As for :meth:`SerializedValue.load`.
        """
        return self.invocation.kwargs.load(ctx)

    def load_state_before(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the instance state captured before the call.

        Raises ``ValueError`` when the record is not a method recording.

        Parameters
        ----------
        ctx : DeserializationContext
            As for :meth:`load_output`.

        Returns
        -------
        Any
            The recorded state - a dict of attribute name to value.

        Raises
        ------
        ValueError
            If the record has no ``state_before``.
        DeserializationError, IdentityResolutionError
            As for :meth:`SerializedValue.load`.
        """
        if self.state_before is None:
            raise ValueError("this record has no state_before (not a method recording)")
        return self.state_before.load(ctx)

    def load_state_after(self, ctx: DeserializationContext) -> Any:
        """Reconstruct the instance state captured after the call.

        Raises ``ValueError`` when the record is not a method recording.

        Parameters
        ----------
        ctx : DeserializationContext
            As for :meth:`load_output`.

        Returns
        -------
        Any
            The recorded state - a dict of attribute name to value.

        Raises
        ------
        ValueError
            If the record has no ``state_after``.
        DeserializationError, IdentityResolutionError
            As for :meth:`SerializedValue.load`.
        """
        if self.state_after is None:
            raise ValueError("this record has no state_after (not a method recording)")
        return self.state_after.load(ctx)

    def to_dict(self) -> dict:
        """Convert to a plain JSON-safe dict (the on-disk record format).

        Optional fields are only present as keys when set, so a ``None`` Python
        return value (``output.data is None``) can be told apart from "no output was
        recorded" (``"output" not in dict``, e.g. because the call raised).

        Returns
        -------
        dict
            Always the keys ``schema``, ``record_id``, ``input_hash``, ``created_at``,
            ``identity``, ``invocation`` and ``tags``; ``output``, ``exception``,
            ``state_before`` and ``state_after`` only when set. The serialized trees
            and ``tags`` are shared with the record, not copied.
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

        Parameters
        ----------
        d : dict
            A record as parsed from its JSON file. ``tags`` may be absent and defaults
            to an empty dict.

        Returns
        -------
        RegressionRecord
            The record, with no side file loaded yet.

        Raises
        ------
        CorruptRecordError
            If the record was written by a newer format than this version reads.
        KeyError, TypeError
            If a required key is missing, or a nested mapping has fields its dataclass
            does not accept. :class:`~regression_records.store.RecordStore` reports
            both as :class:`~regression_records.exceptions.CorruptRecordError`.
        """
        schema = SchemaInfo(**d["schema"])
        if schema.schema_version > SCHEMA_VERSION:
            # Reading a future format as if it were this one would compare against
            # fields whose meaning may have changed - a wrong answer, not an error.
            raise CorruptRecordError(
                f"record uses schema version {schema.schema_version} (written by "
                f"regression-records {schema.library_version}), but this version reads "
                f"up to {SCHEMA_VERSION}; upgrade regression-records to load it"
            )
        identity = FunctionIdentity(**d["identity"])
        invocation = Invocation(
            args=SerializedValue(d["invocation"]["args"]),
            kwargs=SerializedValue(d["invocation"]["kwargs"]),
        )
        return cls(
            schema=schema,
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
