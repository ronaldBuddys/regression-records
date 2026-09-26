"""Define "an object stored as its attributes", for recording, comparison and replay.

An instance of an ordinary class is recorded as its class's import path plus its
attributes - every entry of its ``__dict__`` and every slot set anywhere in its MRO - and
rebuilt by importing the class, allocating it with ``object.__new__`` and setting those
attributes back. Methods are never stored: they come from whatever the class is when the
record is loaded, which is the point of replaying against new code.

Serialization, comparison, method-state capture and receiver rebuild all use these
functions, so they cannot disagree on what an object consists of: anything recorded by
its attributes is also compared, snapshotted and rebuilt by them.

``__getstate__`` / ``__setstate__`` are deliberately ignored. Honouring them would make a
record depend on per-class code, and couple it to that code as it was when recorded.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib
import pathlib
import types
from typing import Any

from .exceptions import (
    DeserializationError,
    IdentityResolutionError,
    SerializationError,
)

#: Values that are never objects to record by attributes: they describe code, not data.
_NOT_DATA = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.ModuleType,
    types.GeneratorType,
    types.CoroutineType,
    types.AsyncGeneratorType,
)

#: Python-level stdlib classes that leaf serializers own (``EnumSerializer``,
#: ``PathSerializer``), listed here so comparison and method capture agree. Stored by
#: attributes they would record CPython internals that differ between Python versions,
#: and an Enum member must load as the member itself, not a copy of it.
_LEAF_CLASSES = (enum.Enum, pathlib.PurePath)

_NEVER_BY_ATTRIBUTES = _NOT_DATA + _LEAF_CLASSES

#: A slot every class may declare that holds no data of the instance's own.
_IGNORED_SLOTS = frozenset({"__weakref__"})

#: ``Py_TPFLAGS_IMMUTABLETYPE``: set on every class implemented in C (static or heap
#: type), never on one created by a ``class`` statement.
_C_CLASS_FLAG = 1 << 8


def class_name(cls: type) -> str:
    """Return ``module.qualname`` for ``cls``, the name records and messages use."""
    return f"{cls.__module__}.{cls.__qualname__}"


def is_namedtuple_class(cls: type) -> bool:
    """Whether ``cls`` is a ``collections.namedtuple`` / ``NamedTuple`` class."""
    return issubclass(cls, tuple) and hasattr(cls, "_fields")


def is_namedtuple(obj: Any) -> bool:
    """Whether ``obj`` is an instance of a ``collections.namedtuple`` / ``NamedTuple``."""
    return is_namedtuple_class(type(obj))


def _adds_c_state(klass: type) -> bool:
    """Whether ``klass`` is a C class that adds instance state below Python attributes.

    A C class may contribute behaviour and still leave an instance fully described by
    its attributes (``object`` itself, ``typing.Generic`` on some versions). One that
    enlarges the instance layout or declares members (``_io.TextIOWrapper``,
    ``BaseException``) holds state no attribute exposes.
    """
    if not klass.__flags__ & _C_CLASS_FLAG:
        return False
    members = any(
        isinstance(v, types.MemberDescriptorType) for v in vars(klass).values()
    )
    return members or klass.__basicsize__ != object.__basicsize__ or klass.__itemsize__


@functools.cache
def _rebuildable(cls: type) -> bool:
    """Whether an instance of ``cls`` is fully described by, and rebuilt from, attributes.

    Two conditions, both about C-level state no attribute exposes. No class in the MRO
    is a C class holding such state. And ``object.__new__(cls)`` allocates an instance,
    as rebuilding one will: it rejects any class with its own C-level constructor
    (datetimes, locks, mappingproxy). Allocation has no other side effect, so trying it
    is the exact test.
    """
    if any(_adds_c_state(klass) for klass in cls.__mro__):
        return False
    try:
        object.__new__(cls)
    except TypeError:
        return False
    return True


def stored_by_attributes(obj: Any) -> bool:
    """Decide whether ``obj`` is recorded, compared and rebuilt by its attributes.

    Parameters
    ----------
    obj : Any
        A value no leaf serializer or comparator claimed.

    Returns
    -------
    bool
        True for a named tuple, or an instance of a class whose whole state is its
        attributes. False for classes, functions, modules, generators, the stdlib
        classes leaf serializers own (``Enum``, ``pathlib`` paths) and C types holding
        state of their own (file handles, locks, datetimes, compiled-extension objects).
    """
    if is_namedtuple(obj):
        return True
    if isinstance(obj, _NEVER_BY_ATTRIBUTES):
        return False
    return _rebuildable(type(obj))


def require_stored_by_attributes(obj: Any, doing: str) -> None:
    """Raise unless ``obj`` can be recorded by its attributes.

    Parameters
    ----------
    obj : Any
        The value about to be recorded.
    doing : str
        What was being attempted, for the message: ``"record a value"``, say.

    Raises
    ------
    SerializationError
        If :func:`stored_by_attributes` is false for ``obj``.
    """
    if not stored_by_attributes(obj):
        raise SerializationError(
            f"cannot {doing} of type '{class_name(type(obj))}': it is not a supported "
            "leaf type, and it cannot be rebuilt from its attributes (functions, "
            "classes, modules, and C types such as locks, file handles or "
            "compiled-extension objects). Record the data it holds instead, or "
            "register a leaf Serializer for the type."
        )


@functools.cache
def _slot_descriptors(cls: type) -> dict[str, types.MemberDescriptorType]:
    """Return every slot ``cls`` and its bases declare, by attribute name.

    Reading the member descriptors rather than ``__slots__`` gives each slot's real,
    name-mangled attribute name (``_Vec__x`` for ``__x``) with no mangling of our own.
    """
    return {
        name: value
        for klass in reversed(cls.__mro__)
        for name, value in vars(klass).items()
        if isinstance(value, types.MemberDescriptorType) and name not in _IGNORED_SLOTS
    }


def attributes(obj: Any) -> dict[str, Any]:
    """Return ``obj``'s attributes: its ``__dict__`` plus every slot that is set.

    Parameters
    ----------
    obj : Any
        A value for which :func:`stored_by_attributes` is true.

    Returns
    -------
    dict[str, Any]
        For a named tuple, its fields by name. Otherwise the instance ``__dict__``
        (when it has one) merged with each set slot of its class and bases; an unset
        slot is absent. A new dict: changing it does not change ``obj``.
    """
    cls = type(obj)
    if is_namedtuple_class(cls):
        return dict(zip(cls._fields, obj, strict=True))
    found = dict(vars(obj)) if hasattr(obj, "__dict__") else {}
    for name, descriptor in _slot_descriptors(cls).items():
        try:
            found[name] = descriptor.__get__(obj, cls)
        except AttributeError:
            continue  # a slot that was never assigned
    return found


def set_attribute(obj: Any, name: str, value: Any) -> None:
    """Set one attribute on ``obj`` as :func:`restore` does, bypassing ``__setattr__``.

    A slot is written through its descriptor and anything else into ``__dict__``, so a
    frozen dataclass, a property of the same name or a validating ``__setattr__`` does
    not intervene: the attribute becomes exactly what was recorded.

    Raises
    ------
    DeserializationError
        If ``name`` is neither a slot of the class nor settable in an instance
        ``__dict__`` - the attribute was renamed or removed since it was recorded.
    """
    cls = type(obj)
    descriptor = _slot_descriptors(cls).get(name)
    if descriptor is not None:
        descriptor.__set__(obj, value)
    elif hasattr(obj, "__dict__"):
        obj.__dict__[name] = value
    else:
        raise DeserializationError(
            f"recorded attribute {name!r} no longer exists on '{class_name(cls)}' (a "
            "slot renamed or removed since the record was written); re-record it"
        )


def restore(cls: type, attrs: dict[str, Any]) -> Any:
    """Rebuild an instance of ``cls`` from its recorded attributes, without ``__init__``.

    Parameters
    ----------
    cls : type
        The class, already imported.
    attrs : dict[str, Any]
        The attributes, already loaded.

    Returns
    -------
    Any
        A new instance with exactly those attributes. Neither ``__init__`` nor
        ``__post_init__`` runs, so an invariant only they establish is not checked.

    Raises
    ------
    DeserializationError
        If the recorded attributes no longer fit the class: a named tuple's fields or a
        dataclass's fields changed, or a slot was renamed or removed.
    """
    if is_namedtuple_class(cls):
        added = [name for name in cls._fields if name not in attrs]
        removed = [name for name in attrs if name not in cls._fields]
        if added or removed:
            raise DeserializationError(
                f"the fields of '{class_name(cls)}' changed since the record was "
                f"written (added {added}, removed {removed}); re-record it"
            )
        # tuple.__new__ rather than cls(*items): the class is named by a record file,
        # and calling it would run whatever constructor that name resolves to.
        return tuple.__new__(cls, [attrs[field] for field in cls._fields])
    if not _rebuildable(cls):
        raise DeserializationError(
            f"'{class_name(cls)}' can no longer be rebuilt from its attributes: "
            "object.__new__ refuses it"
        )
    if dataclasses.is_dataclass(cls):
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in attrs]
        if missing:
            raise DeserializationError(
                f"'{class_name(cls)}' has field(s) {missing} that the record does not "
                "hold (added since it was written); re-record it"
            )
    obj = object.__new__(cls)
    for name, value in attrs.items():
        set_attribute(obj, name, value)
    return obj


def class_reference(cls: type) -> dict[str, str]:
    """Return the ``{"module", "qualname"}`` a record uses to name ``cls``.

    Raises
    ------
    SerializationError
        If ``cls`` could not be imported by that name later: it is defined in
        ``__main__`` or inside a function. Refused now rather than at load time, when
        the record already exists and can never be read.
    """
    if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__:
        where = (
            "'__main__'" if cls.__module__ == "__main__" else "the body of a function"
        )
        raise SerializationError(
            f"cannot serialize instances of class '{cls.__qualname__}' defined in "
            f"{where} - it cannot be re-imported by name from a fresh process "
            "later. Move the class to the top level of a regular importable module."
        )
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def import_class(class_info: dict[str, str]) -> type:
    """Import the class a record names.

    Raises
    ------
    IdentityResolutionError
        If the module or attribute no longer exists, or the name resolves to something
        other than a class. The names come from a record file, so nothing but a class
        is ever handed back to be instantiated.
    """
    try:
        obj: Any = importlib.import_module(class_info["module"])
        for part in class_info["qualname"].split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as e:
        raise IdentityResolutionError(
            f"cannot import class '{class_info['qualname']}' from module "
            f"'{class_info['module']}': {e}"
        ) from e
    if not isinstance(obj, type):
        raise IdentityResolutionError(
            f"'{class_info['module']}.{class_info['qualname']}' is not a class"
        )
    return obj
