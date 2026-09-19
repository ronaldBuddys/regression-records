"""What "an object described by its attribute state" means, for recording *and*
comparison.

Both the generic serializer and the generic comparator need this, and they have to agree
exactly: anything that can be *recorded* by its state must also be *compared* by its
state, or a record would be written in a form the comparator refuses to read. Keeping
one definition here is what makes that agreement structural rather than a promise in two
docstrings.
"""

from __future__ import annotations

from typing import Any, Optional


def has_inspectable_state(obj: Any) -> bool:
    """Whether ``obj`` exposes attribute state at all.

    Note this is true of almost everything on Python 3.11+, which gives every object an
    inherited ``__getstate__`` - callers that dispatch on it are expected to have ruled
    out the types they handle more specifically first.
    """
    return hasattr(obj, "__getstate__") or hasattr(obj, "__dict__")


def object_state(obj: Any) -> Optional[dict]:
    """Extract ``obj``'s state as a plain dict, or ``None`` if it has none to offer.

    ``__getstate__`` wins where it exists, since that is the object's own statement
    about what it consists of - including when it answers ``None``. Python's default
    implementation answers ``None`` for anything whose ``__dict__`` is empty, which is
    what keeps functions and lambdas out: they have a ``__dict__``, but nothing in it
    describes what they do, so recording one would capture nothing while appearing to
    succeed.

    ``None`` is also the answer when the state isn't a dict at all (a ``__slots__``-only
    class yields a tuple). Such objects can be neither recorded nor diffed
    attribute-by-attribute.
    """
    state = obj.__getstate__() if hasattr(obj, "__getstate__") else getattr(obj, "__dict__", None)
    return state if isinstance(state, dict) else None
