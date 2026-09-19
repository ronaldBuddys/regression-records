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
    """

    type_tag: str

    def can_handle(self, obj: Any) -> bool:
        """Return True if this serializer knows how to store ``obj``."""
        ...

    def store(self, obj: Any, ctx: SerializationContext) -> dict:
        """Return a JSON-safe payload describing ``obj`` (excluding the type tag)."""
        ...

    def load(self, blob: dict, ctx: DeserializationContext) -> Any:
        """Reconstruct the original object from a payload previously returned by ``store``."""
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
        """Return the first registered serializer that can handle ``obj``, or None."""
        for serializer in self._serializers:
            if serializer.can_handle(obj):
                return serializer
        return None

    def by_tag(self, tag: str) -> Serializer:
        """Look up a serializer by its ``type_tag`` (used when loading)."""
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
        """How many buffered writes are waiting for :meth:`flush` (0 unless deferred)."""
        return len(self._pending)

    def flush(self) -> list[Path]:
        """Perform every buffered write and return the paths written."""
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
        """Drop every buffered write without performing it."""
        self._pending.clear()

    def ref(self, filename: str) -> str:
        """Build the reference string a serializer should store as ``path``.

        Combines ``filename`` with ``ref_prefix`` so it resolves correctly relative to
        the directory a matching :class:`DeserializationContext` will use as
        ``base_dir`` (e.g. ``"../data/"`` when data files live in a sibling directory
        of the records that reference them).
        """
        return f"{self.ref_prefix}{filename}"

    def write_if_absent(self, filename: str, write_fn: Callable[[Path], None]) -> Path:
        """Write ``filename`` under ``data_dir`` via ``write_fn(path)``, unless it exists.

        Because filenames are content-addressed, an existing file with this name is
        assumed to already hold identical content.

        When the context is ``deferred``, the write is buffered until :meth:`flush`;
        the returned path is the same either way, so serializers need not care.
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
    """

    base_dir: Path
    registry: SerializerRegistry
    #: The directory every reference must resolve inside; ``base_dir`` when not given.
    #: A store sets it to the identity directory, since its records sit in
    #: ``records/`` and refer to ``../data/``.
    root: Path | None = None

    def __post_init__(self) -> None:
        """Coerce the directories to ``Path``, so callers may pass strings."""
        self.base_dir = Path(self.base_dir)
        self.root = Path(self.root) if self.root is not None else self.base_dir

    def resolve(self, relative_path: str) -> Path:
        """Resolve a relative file reference stored in a record against ``base_dir``.

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
