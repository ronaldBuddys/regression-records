"""Define "an object described by its attribute state", for recording and comparison.

Both the generic serializer and the generic comparator need this, and they have to agree
exactly: anything that can be *recorded* by its state must also be *compared* by its
state, or a record would be written in a form the comparator refuses to read. Keeping
one definition here is what makes that agreement structural rather than a promise in two
docstrings.
"""

from __future__ import annotations

from typing import Any


def object_state(obj: Any) -> dict | None:
    """Extract ``obj``'s state as a plain dict, or ``None`` if it has none to offer.

    ``__getstate__`` is the object's own statement about what it consists of, and every
    object has one: a class's own, or the default inherited from ``object`` (Python
    3.11+, which is why that is the package's floor). The default answers ``None`` for
    anything whose ``__dict__`` is empty, which is what keeps functions and lambdas
    out: they have a ``__dict__``, but nothing in it describes what they do, so
    recording one would capture nothing while appearing to succeed.

    ``None`` is also the answer when the state isn't a dict at all (a ``__slots__``-only
    class yields a tuple). Such objects can be neither recorded nor diffed
    attribute-by-attribute.
    """
    if isinstance(obj, type):
        # A class has __getstate__ too, but unbound: calling it raises TypeError. A
        # class passed as a value has no instance state to describe in any case.
        return None
    state = obj.__getstate__()
    return state if isinstance(state, dict) else None


def is_namedtuple(obj: Any) -> bool:
    """Whether ``obj`` is an instance of a ``collections.namedtuple`` / ``NamedTuple``.

    Here for the same reason as :func:`object_state`: the serializer that records a
    named tuple with its class and the comparison that treats it as its own kind of
    sequence have to agree on what one is.
    """
    return isinstance(obj, tuple) and hasattr(type(obj), "_fields")
