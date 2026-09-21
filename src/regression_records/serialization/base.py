"""The extensible serializer protocol and registry.

A :class:`Serializer` handles one Python type that is inconvenient to represent
directly as JSON (arrays, dataframes, arbitrary objects). Serializers are looked up
through a :class:`SerializerRegistry`, so new types can be supported without touching
the core tree-walking logic in :mod:`regression_records.serialization`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .._atomic import atomic_write
from ..exceptions import CorruptRecordError, DeserializationError


@runtime_checkable
class Serializer(Protocol):
    """Handles storing/loading one externally-representable Python type.

    ``type_tag`` is written into the JSON tree (as the reserved ``"__type__"`` key) so
    that ``load`` can later be dispatched back to the same serializer.

    Attributes
    ----------
    type_tag : str
        The name this serializer's values are tagged with in a record. Unique within a
        registry, and part of the on-disk format: changing it orphans existing records.
    """

    type_tag: str

    def can_handle(self, obj: Any) -> bool:
        """Return True if this serializer knows how to store ``obj``.

        Parameters
        ----------
        obj : Any
            A value the tree walk does not record as plain JSON itself.

        Returns
        -------
        bool
            True to have :meth:`store` called with ``obj``. The first serializer in the
            registry to say so wins.
        """
        ...

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return a JSON-safe payload describing ``obj`` (excluding the type tag).

        Parameters
        ----------
        obj : Any
            A value :meth:`can_handle` claimed.
        ctx : SerializationContext
            Where to write side files (:meth:`SerializationContext.write_if_absent`)
            and how to refer to them (:meth:`SerializationContext.ref`).

        Returns
        -------
        dict
            String-keyed and JSON-safe, without the reserved ``"__type__"`` key - the
            caller adds that from ``type_tag``. Nested values that are not JSON-safe
            themselves go through
            :func:`~regression_records.serialization.to_json_tree` first.

        Raises
        ------
        SerializationError
            By convention, if ``obj`` cannot be recorded faithfully.
        """
        ...

    def load(self, blob: dict, ctx: DeserializationContext) -> Any:
        """Reconstruct the original object from a payload previously returned by ``store``.

        Parameters
        ----------
        blob : dict
            The payload :meth:`store` returned, as read back from JSON, with the
            ``"__type__"`` key added.
        ctx : DeserializationContext
            Turns stored file references into paths
            (:meth:`DeserializationContext.resolve`).

        Returns
        -------
        Any
            A value that compares equal to the one stored.

        Raises
        ------
        DeserializationError
            By convention, if the payload or a file it references cannot be read.
        """
        ...


class SerializerRegistry:
    """An ordered collection of serializers, searched first-match-wins."""

    def __init__(self) -> None:
        self._serializers: list[Serializer] = []
        self._fallbacks = 0
        self._by_tag: dict[str, Serializer] = {}

    def register(
        self,
        serializer: Serializer,
        *,
        priority: int | None = None,
        fallback: bool = False,
        replace: bool = False,
    ) -> None:
        """Add ``serializer`` to the search order.

        By default it goes last among the ordinary serializers but *before* any
        registered fallback, so a catch-all registered with ``fallback=True`` keeps
        getting last refusal however many serializers are added after it. ``priority``
        is an explicit list index for the rare case that isn't good enough.

        With ``replace=True`` it takes over the position of the serializer already
        registered under its ``type_tag`` - the way to swap in a differently
        configured instance of a built-in.

        Parameters
        ----------
        serializer : Serializer
            The serializer to add.
        priority : int, optional
            The list index to insert at, with ``list.insert`` semantics (``0`` is
            consulted first). When omitted: the end of the list for a fallback,
            otherwise just before the first fallback. Ignored when replacing.
        fallback : bool, default False
            Mark ``serializer`` as a catch-all that later default registrations are
            placed ahead of. Ignored when replacing.
        replace : bool, default False
            Allow ``serializer`` to take the place of the one already registered under
            its ``type_tag``. Without such a serializer it is an ordinary registration.

        Raises
        ------
        ValueError
            If ``type_tag`` is already registered and ``replace`` is false. Values are
            *written* by the first serializer to claim them but *read* by tag, so two
            serializers sharing a tag would have one write what the other reads.
        """
        existing = self._by_tag.get(serializer.type_tag)
        if existing is not None:
            if not replace:
                raise ValueError(
                    f"a serializer is already registered for type tag "
                    f"{serializer.type_tag!r} ({type(existing).__name__}); pass "
                    "replace=True to swap it out, or choose a different type_tag"
                )
            self._serializers[self._serializers.index(existing)] = serializer
            self._by_tag[serializer.type_tag] = serializer
            return
        if priority is None:
            priority = (
                len(self._serializers)
                if fallback
                else len(self._serializers) - self._fallbacks
            )
        self._serializers.insert(priority, serializer)
        if fallback:
            self._fallbacks += 1
        self._by_tag[serializer.type_tag] = serializer

    def copy(self) -> SerializerRegistry:
        """Return a new registry with the same serializers, in the same order.

        The way to add a type to *one* decoration: copy the default registry, register
        on the copy, and inject it. Registering on ``default_registry`` itself changes
        the behavior of every decoration in the process, which is rarely what a library
        or a single test wants.

        The serializers themselves are shared, not copied - they are stateless
        dispatchers, and duplicating them would only make the same type answer to two
        objects.

        Returns
        -------
        SerializerRegistry
            An independent registry: registering on it leaves this one unchanged.
        """
        clone = SerializerRegistry()
        clone._serializers = list(self._serializers)
        clone._fallbacks = self._fallbacks
        clone._by_tag = dict(self._by_tag)
        return clone

    def __len__(self) -> int:
        """Return the number of registered serializers, fallbacks included."""
        return len(self._serializers)

    def find(self, obj: Any) -> Serializer | None:
        """Return the first registered serializer that can handle ``obj``, or None.

        Parameters
        ----------
        obj : Any
            The value to be stored.

        Returns
        -------
        Serializer or None
            The first serializer, in search order, whose ``can_handle(obj)`` is true.
        """
        for serializer in self._serializers:
            if serializer.can_handle(obj):
                return serializer
        return None

    def by_tag(self, tag: str) -> Serializer:
        """Look up a serializer by its ``type_tag`` (used when loading).

        Parameters
        ----------
        tag : str
            The ``"__type__"`` value read from a record.

        Returns
        -------
        Serializer
            The serializer registered under ``tag``.

        Raises
        ------
        DeserializationError
            If no serializer is registered under ``tag``.
        """
        try:
            return self._by_tag[tag]
        except KeyError:
            raise DeserializationError(
                f"no serializer is registered for type tag {tag!r}; the record may have "
                "been written by a version of regression-records with different "
                "serializer support"
            ) from None


@dataclass
class SerializationContext:
    """Everything a :class:`Serializer` needs while storing a value.

    ``data_dir`` is where side files (``.npy``, ``.parquet``, ...) are written.

    With ``deferred=True`` those writes are buffered instead of performed immediately,
    and only happen on :meth:`flush`. This exists because a value often has to be
    serialized *before* it is known whether it will be kept: computing an invocation's
    ``input_hash`` requires the serialized argument tree, but in validate-only mode - or
    on a mismatch under ``OverwritePolicy.NEVER`` - no record is written, and eager
    writes would leave unreferenced files behind in ``data_dir``. Call :meth:`flush` once
    the record is committed, or :meth:`discard` to drop the buffered writes.

    Attributes
    ----------
    data_dir : Path
        The directory side files are written to; created on the first write. A string
        is accepted and converted.
    registry : SerializerRegistry
        The serializers values are dispatched to.
    ref_prefix : str, default ""
        Prepended to a filename by :meth:`ref`: the relative path, ending in ``/``,
        from the directory holding the record's JSON to ``data_dir``.
    deferred : bool, default False
        Buffer side-file writes until :meth:`flush` instead of performing them at once.
    """

    data_dir: Path
    registry: SerializerRegistry
    ref_prefix: str = ""
    deferred: bool = False
    _pending: dict[Path, Callable[[Path], None]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    #: ``id()`` of every container currently being converted, for cycle detection.
    active: set[int] = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Coerce ``data_dir`` to a ``Path``, so callers may pass a string."""
        self.data_dir = Path(self.data_dir)

    @property
    def pending_writes(self) -> int:
        """How many buffered writes are waiting for :meth:`flush` (0 unless deferred).

        Returns
        -------
        int
            The number of distinct files buffered.
        """
        return len(self._pending)

    def flush(self) -> list[Path]:
        """Perform every buffered write and return the paths written.

        Returns
        -------
        list[Path]
            The files this call created, in the order they were buffered. A file that
            came to exist in the meantime is skipped and not listed. Empty when
            nothing was buffered.

        Raises
        ------
        OSError
            If ``data_dir`` cannot be created or a file cannot be written, as raised by
            the buffered write function. The writes not yet performed stay buffered.
        """
        written = []
        if self._pending:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        for path, write_fn in self._pending.items():
            if not path.exists():
                atomic_write(path, write_fn)
                written.append(path)
        self._pending.clear()
        return written

    def discard(self) -> None:
        """Drop every buffered write without performing it.

        Files already on disk are left alone; only the buffer is cleared.
        """
        self._pending.clear()

    def ref(self, filename: str) -> str:
        """Build the reference string a serializer should store as ``path``.

        Combines ``filename`` with ``ref_prefix`` so it resolves correctly relative to
        the directory a matching :class:`DeserializationContext` will use as
        ``base_dir`` (e.g. ``"../data/"`` when data files live in a sibling directory
        of the records that reference them).

        Parameters
        ----------
        filename : str
            The name of a file under ``data_dir``, as passed to
            :meth:`write_if_absent`.

        Returns
        -------
        str
            ``ref_prefix`` followed by ``filename``, for
            :meth:`DeserializationContext.resolve` to turn back into a path.
        """
        return f"{self.ref_prefix}{filename}"

    def write_if_absent(self, filename: str, write_fn: Callable[[Path], None]) -> Path:
        """Write ``filename`` under ``data_dir`` via ``write_fn(path)``, unless it exists.

        Because filenames are content-addressed, an existing file with this name is
        assumed to already hold identical content.

        When the context is ``deferred``, the write is buffered until :meth:`flush`;
        the returned path is the same either way, so serializers need not care.

        Parameters
        ----------
        filename : str
            The file's name within ``data_dir`` - content-addressed, see
            :func:`~regression_records.serialization.naming.content_filename`.
        write_fn : Callable[[Path], None]
            Writes the content to the path it is given. That path is a temporary
            sibling with the same suffix, moved into place once ``write_fn`` returns,
            so an interrupted write never leaves a file under its final name.

        Returns
        -------
        Path
            ``data_dir / filename``, whether or not the file exists yet.

        Raises
        ------
        OSError
            When not deferred: if ``data_dir`` cannot be created or the file cannot be
            written, as raised by ``write_fn``.
        """
        path = self.data_dir / filename
        if self.deferred:
            if not path.exists():
                self._pending.setdefault(path, write_fn)
            return path
        if not path.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(path, write_fn)
        return path


@dataclass
class DeserializationContext:
    """Everything a :class:`Serializer` needs while loading a value.

    ``base_dir`` is the directory that relative ``path`` references are resolved
    against (the directory containing the record's JSON file).

    Attributes
    ----------
    base_dir : Path
        The directory stored references are relative to. A string is accepted and
        converted.
    registry : SerializerRegistry
        The serializers tagged values are dispatched to, by ``type_tag``.
    root : Path, optional
        The directory every reference must resolve inside; ``base_dir`` when not
        given. A store sets it to the identity directory, since its records sit in
        ``records/`` and refer to ``../data/``.
    verify : bool, default False
        Check each content-addressed side file against the hash in its name as it is
        loaded (see :func:`~regression_records.serialization.naming.verify_content_file`),
        so a modified or swapped file raises ``CorruptRecordError`` instead of loading
        as the recorded value. Costs one extra hash of every file read.
    """

    base_dir: Path
    registry: SerializerRegistry
    root: Path | None = None
    verify: bool = False

    def __post_init__(self) -> None:
        """Coerce the directories to ``Path``, so callers may pass strings."""
        self.base_dir = Path(self.base_dir)
        self.root = Path(self.root) if self.root is not None else self.base_dir

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative file reference stored in a record against ``base_dir``.

        Parameters
        ----------
        relative_path : str
            A reference as built by :meth:`SerializationContext.ref`, e.g.
            ``"../data/arr-0123abcd.npy"``.

        Returns
        -------
        Path
            The absolute path, with symlinks and ``..`` resolved. The file is not
            checked for existence.

        Raises
        ------
        CorruptRecordError
            If the reference is absolute or resolves outside :attr:`root`. The path
            comes from a record file, and whatever it names is handed to a file
            reader, so a record must not be able to point one at an arbitrary file.
            (An absolute path needs rejecting by itself: joining one to ``base_dir``
            discards ``base_dir`` entirely.)
        """
        resolved = (self.base_dir / relative_path).resolve()
        if Path(relative_path).is_absolute() or not resolved.is_relative_to(
            self.root.resolve()
        ):
            raise CorruptRecordError(
                f"record references a file outside its store: {relative_path!r}"
            )
        return resolved
