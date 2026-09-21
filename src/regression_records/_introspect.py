"""Pulling apart a method call: which arguments identify it, and what state it acts on.

This is the only thing ``record_method`` needs that ``record_func`` doesn't. A method's
result depends on its receiver as much as on its arguments, so the receiver is recorded
as *state* rather than as an argument, snapshotted on both sides of the call.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._object_state import object_state
from .exceptions import SerializationError


def split_receiver(
    bound: inspect.BoundArguments,
) -> tuple[Any, list, Mapping[str, Any]]:
    """Separate a bound method call's receiver from the arguments that identify it.

    ``self`` is deliberately not part of the recorded arguments, and so not part of the
    input hash: the object is described by its captured state instead. Hashing it as an
    argument would make every instance - and every run, since most objects hash by
    identity - a different case.

    Remaining arguments are recorded by name. Once the receiver is gone, positional
    order no longer describes the call, and ``apply_defaults()`` has already named
    everything anyway.
    """
    parameters = list(bound.arguments)
    if not parameters:
        raise TypeError(
            "record_method expected a method taking a receiver (self/cls) as its first "
            "parameter; use record_func for a plain function"
        )
    receiver_name = parameters[0]
    instance = bound.arguments[receiver_name]
    arguments = {
        name: value for name, value in bound.arguments.items() if name != receiver_name
    }
    return instance, [], arguments


@dataclass(frozen=True)
class StateSpec:
    """Which of a receiver's attributes are recorded as its state."""

    #: Attribute names to record, or None for "every attribute the object exposes".
    include: tuple[str, ...] | None = None
    #: Attribute names to leave out - useful for caches, clients, and file handles.
    exclude: tuple[str, ...] = ()

    @classmethod
    def build(
        cls, include: Sequence[str] | None = None, exclude: Sequence[str] = ()
    ) -> StateSpec:
        return cls(
            include=tuple(include) if include is not None else None,
            exclude=tuple(exclude),
        )

    def selects(self, name: str) -> bool:
        if name in self.exclude:
            return False
        return self.include is None or name in self.include


def capture_state(instance: Any, spec: StateSpec) -> dict:
    """Snapshot ``instance``'s selected attributes, deep-copied.

    The copy is the whole point on the *before* side: the method is about to mutate this
    object, and a snapshot that shared its containers would quietly become a picture of
    the state afterwards. Copying both sides keeps one code path, and costs far less than
    the serialization that follows it.
    """
    state = object_state(instance)
    if state is None and hasattr(instance, "__dict__") and not vars(instance):
        # A normal object that simply has no attributes set yet. "Nothing assigned" is a
        # legitimate state for a method to start from, and differs from "this type has no
        # inspectable state at all", which is what the error below is about.
        state = {}
    if state is None:
        raise SerializationError(
            f"cannot capture state for an instance of '{type(instance).__qualname__}': it "
            "exposes no dict-shaped attribute state (a __slots__-only class, or a "
            "__getstate__() returning something other than a dict). Record it with a "
            "custom Serializer, or use record_func on a plain function instead."
        )

    selected = {name: value for name, value in state.items() if spec.selects(name)}
    try:
        return copy.deepcopy(selected)
    except Exception as e:  # noqa: BLE001 - any copy failure gets the same advice
        raise SerializationError(
            f"cannot snapshot state for an instance of '{type(instance).__qualname__}': "
            f"deep-copying its attributes failed ({e}). Exclude the offending attribute "
            "with exclude_attrs=(...), or list only what matters with state_attrs=(...)."
        ) from e
