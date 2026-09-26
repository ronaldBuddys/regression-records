"""The search-order machinery the serializer and comparator registries share.

Both are an ordered list of leaf handlers searched first-match-wins. Only what an entry
is asked - ``can_handle(obj)`` or ``can_handle(expected, actual)`` - and the serializer
registry's lookup by tag differ, so those stay in the subclasses. Neither holds a
catch-all: a value no entry claims is stored and compared by its attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Self, TypeVar

T = TypeVar("T")


class OrderedRegistry(Generic[T]):
    """An ordered collection of entries, searched first-match-wins."""

    def __init__(self) -> None:
        self._entries: list[T] = []

    def __len__(self) -> int:
        """Return the number of registered entries, fallbacks included."""
        return len(self._entries)

    def copy(self) -> Self:
        """Return a new registry with the same entries, in the same order.

        The way to add an entry for one store or one replay: copy the default
        registry, register on the copy, and inject it. Registering on a default
        registry changes the behaviour of everything in the process, which is rarely
        what a library or a single test wants.

        Returns
        -------
        Self
            An independent registry: registering on it leaves this one unchanged. The
            entries themselves are shared, not copied - they are stateless dispatchers.
        """
        clone = type(self)()
        clone._entries = list(self._entries)
        return clone

    def _insert(self, entry: T, priority: int | None) -> None:
        """Insert ``entry`` at list index ``priority``, or at the end when None.

        Raises ``ValueError`` if ``priority`` is out of range.
        """
        size = len(self._entries)
        if priority is None:
            priority = size
        if not 0 <= priority <= size:
            raise ValueError(
                f"priority {priority} is out of range: it must be between 0 and {size}"
            )
        self._entries.insert(priority, entry)

    def _replace(self, old: T, new: T) -> None:
        """Put ``new`` in ``old``'s place in the search order."""
        self._entries[self._entries.index(old)] = new

    def _first(self, claims: Callable[[T], bool]) -> T | None:
        """Return the first entry, in search order, for which ``claims`` is true."""
        return next((entry for entry in self._entries if claims(entry)), None)
