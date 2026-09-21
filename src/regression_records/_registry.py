"""The search-order machinery the serializer and comparator registries share.

Both are an ordered list searched first-match-wins: ordinary entries, then fallbacks
(catch-alls that keep last refusal however many entries are added after them). Only
what an entry is asked - ``can_handle(obj)`` or ``can_handle(expected, actual)`` - and
the serializer registry's lookup by tag differ, so those stay in the subclasses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Self, TypeVar

T = TypeVar("T")


def insertion_index(
    size: int, fallbacks: int, priority: int | None, *, fallback: bool
) -> int:
    """Return the list index to insert a new registry entry at.

    A registry's list is its ordinary entries followed by its fallbacks, and an entry
    must stay in its own segment: an ordinary entry placed after a fallback would never
    be consulted for anything the fallback claims, and the fallback count used for later
    default positions would be wrong.

    Parameters
    ----------
    size : int
        How many entries the registry holds, fallbacks included.
    fallbacks : int
        How many of them are fallbacks (the last ``fallbacks`` entries).
    priority : int or None
        The requested index (``0`` is consulted first), or None for the default: the
        end of the entry's segment.
    fallback : bool
        Whether the new entry is a fallback.

    Returns
    -------
    int
        An index within the entry's segment.

    Raises
    ------
    ValueError
        If ``priority`` is outside the entry's segment.
    """
    ordinary = size - fallbacks
    low, high = (ordinary, size) if fallback else (0, ordinary)
    if priority is None:
        return high
    if not low <= priority <= high:
        kind = "a fallback" if fallback else "an ordinary entry"
        raise ValueError(
            f"priority {priority} is out of range for {kind}: it must be between "
            f"{low} and {high} (ordinary entries come first, then fallbacks)"
        )
    return priority


class OrderedRegistry(Generic[T]):
    """An ordered collection of entries, ordinary ones before fallbacks."""

    def __init__(self) -> None:
        self._entries: list[T] = []
        self._fallbacks = 0

    def __len__(self) -> int:
        """Return the number of registered entries, fallbacks included."""
        return len(self._entries)

    def copy(self) -> Self:
        """Return a new registry with the same entries, in the same order.

        The way to add an entry for *one* decoration: copy the default registry,
        register on the copy, and inject it. Registering on a default registry changes
        the behaviour of every decoration in the process, which is rarely what a
        library or a single test wants.

        Returns
        -------
        Self
            An independent registry: registering on it leaves this one unchanged. The
            entries themselves are shared, not copied - they are stateless dispatchers.
        """
        clone = type(self)()
        clone._entries = list(self._entries)
        clone._fallbacks = self._fallbacks
        return clone

    def _insert(self, entry: T, priority: int | None, fallback: bool) -> None:
        """Insert ``entry`` in its segment; see :func:`insertion_index`."""
        index = insertion_index(
            len(self._entries), self._fallbacks, priority, fallback=fallback
        )
        self._entries.insert(index, entry)
        if fallback:
            self._fallbacks += 1

    def _replace(self, old: T, new: T) -> None:
        """Put ``new`` in ``old``'s place in the search order."""
        self._entries[self._entries.index(old)] = new

    def _first(self, claims: Callable[[T], bool]) -> T | None:
        """Return the first entry, in search order, for which ``claims`` is true."""
        return next((entry for entry in self._entries if claims(entry)), None)
