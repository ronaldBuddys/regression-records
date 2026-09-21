"""Where a new entry goes in a registry's search order - shared by both registries."""

from __future__ import annotations


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
