"""
Custom classes : stored by their attributes
===========================================

``to_json_tree()`` and ``from_json_tree()`` are the store/load pair the decorators are built
on, and they are usable directly - the successor to the old ``json_store`` /
``load_json_store``.

The headline: **a class needs no registration and no serializer.** Any instance is
recorded as its class's import path plus its attributes - its ``__dict__`` and every slot
set in its class hierarchy - each converted by the same rules, recursively. Loading
imports the class, allocates an instance with ``object.__new__`` and sets those
attributes back. Methods are never stored: they come from the code as it is when the
record is loaded, which is what replaying against new code means.

Serializers exist only for *leaf* values - arrays, frames, datetimes, and C types with no
attributes to record (example 9).
"""

import inspect
import json
import sys
import tempfile
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import numpy as np
from _shared.models import Calibration, Experiment, Point, Quote, Reading, Vec3

from regression_records.serialization import (
    DeserializationContext,
    SerializationContext,
    default_registry,
    from_json_tree,
    to_json_tree,
)

# %%
# The two contexts
# ----------------
#
# A ``SerializationContext`` says where side files go and how they should be referenced; a
# ``DeserializationContext`` says where to resolve those references from. Here the JSON sits
# at the root and its arrays go in ``data/`` beside it, so the reference ``data/arr-....npy``
# resolves from the JSON's own directory - which is what lets the whole directory be moved
# or copied as a unit.

ROOT = Path(tempfile.mkdtemp(prefix="example08-"))

writing = SerializationContext(
    data_dir=ROOT / "data",
    registry=default_registry,
    ref_prefix="data/",
)
reading = DeserializationContext(base_dir=ROOT, registry=default_registry)

# %%
# A plain class
# -------------
#
# ``Point`` has two attributes and nothing else. Its tree is the class's import path plus
# those attributes.

print(json.dumps(to_json_tree(Point(1.0, 2.5), writing), indent=2))

# %%
# Slots, frozen dataclasses and named tuples
# ------------------------------------------
#
# The shape of the class does not matter. ``Vec3`` declares ``__slots__`` and has no
# ``__dict__``; its slots are its attributes. ``Calibration`` is a
# ``@dataclass(frozen=True, slots=True)`` holding a read-only ``MappingProxyType`` - the
# attributes are set back past the frozen ``__setattr__``, and the view comes back as a
# view. A named tuple is stored with its fields by name.

for value in (
    Vec3(1.0, 2.0, 3.0),
    Calibration("rig-2", {"thermocouple-3": 0.05}),
    Quote(bid=99.5, ask=100.5),
):
    tree = to_json_tree(value, writing)
    loaded = from_json_tree(tree, reading)
    print(json.dumps(tree["attrs"]))
    print("   ->", loaded)

# %%
# Nesting
# -------
#
# Serialization recurses, so objects inside objects inside lists all work. This is one
# ``Experiment`` holding a ``Point`` and two ``Reading`` objects, each with an array.

experiment = Experiment(
    name="run-42",
    origin=Point(0.0, 0.0),
    readings=[
        Reading("thermocouple-3", np.array([20.1, 20.4, 20.2])),
        Reading("thermocouple-7", np.array([19.8, 19.9])),
    ],
)

tree = to_json_tree(experiment, writing)
(ROOT / "experiment.json").write_text(json.dumps(tree, indent=2))

print((ROOT / "experiment.json").read_text()[:900], "...")

# %%
# The arrays became references to ``.npy`` files rather than lists of numbers: full
# precision, full dtype, and content-addressed, so two equal arrays anywhere in the tree
# share one file.
#
# Load it back
# ------------
#
# Through a real file, so nothing survives in memory between the two halves.

loaded = from_json_tree(json.loads((ROOT / "experiment.json").read_text()), reading)

print(loaded)
print()
print("type restored:      ", type(loaded) is Experiment)
print(
    "nested type:        ",
    type(loaded.origin) is Point,
    type(loaded.readings[0]) is Reading,
)
print("nested value:       ", loaded.origin == experiment.origin)
print(
    "array restored:     ",
    np.array_equal(loaded.readings[0].samples, np.array([20.1, 20.4, 20.2])),
)
print("array dtype:        ", loaded.readings[0].samples.dtype)

# %%
# What actually happens on load
# -----------------------------
#
# The class is re-imported from the ``module`` / ``qualname`` in the record, allocated with
# ``object.__new__(cls)``, and each recorded attribute is set on it - a slot through its
# descriptor, anything else in ``__dict__``.
#
# Nothing else of the class's runs. ``__init__`` and a dataclass's ``__post_init__`` are
# not called, which is what makes a record loadable when a constructor needs a database, a
# file, or an argument nobody kept; it also means an invariant only they establish is not
# re-established or re-checked. ``__getstate__`` / ``__setstate__`` are not used either:
# an object is recorded as its attributes, whatever its pickling hooks say.
#
# The flip side is that a record names attributes, private ones included. Rename an
# attribute, and records of the class written before the rename no longer fit it; a
# removed slot or an added dataclass field is reported as such when the record is loaded,
# and the record has to be re-recorded.

# %%
# The two rules
# -------------
#
# **1. The class must be importable by name.** It is stored as its module and qualified
# name, so loading a record re-imports it. A class defined in ``__main__`` - a script, or a
# notebook cell - cannot be re-imported from a fresh process, and is refused at store
# time rather than producing a record nobody can read:

from regression_records import SerializationError  # noqa: E402 - kept beside its use


class DefinedRightHere:
    def __init__(self):
        self.x = 1


try:
    to_json_tree(DefinedRightHere(), writing)
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
# This is why the examples keep their classes in ``_shared/models.py``. Move a class into
# any importable module and it records fine.
#
# **2. Every attribute must be data.** An attribute is recorded by the same rules as any
# other value, so one that is not data - an open file, a lock, a function, an object
# from a compiled extension - refuses the whole object. It is never skipped: a record
# that silently lost an attribute would replay as a pass. ``LiveReading`` holds an open
# file:

from _shared.models import LiveReading  # noqa: E402 - kept beside its use

try:
    to_json_tree(LiveReading("thermocouple-3", np.array([20.1])), writing)
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
# Record the data such an object holds instead - here, a plain ``Reading``. For a method
# whose receiver holds one, ``record_method(exclude_attrs=...)`` leaves it out of the
# recorded state (example 6).
#
# The same thing, through a decorator
# -----------------------------------
#
# Everything above is what happens inside ``record_func`` and ``record_method``. Custom
# objects work as arguments, as return values, and as recorded method state, with no
# configuration at all.

from _shared import experiments  # noqa: E402
from _shared.demo import fresh_store, show_record  # noqa: E402

from regression_records import enable  # noqa: E402

STORE = fresh_store(experiments.STORE.name)

# Recording is off unless switched on; this example switches it on from here to the end
# of the script.
enable()

print(inspect.getsource(experiments.recentre))

experiments.recentre(experiment, Point(5.0, 5.0))
show_record(STORE, keys=["output"])

record_text = next(STORE.rglob("rec-*.json")).read_text()
print("\n.npy references in the record:", record_text.count(".npy"))
print(".npy files on disk:           ", len(list(STORE.rglob("*.npy"))))

# %%
# The arguments are stored the same way (the record's ``invocation.args`` holds the same
# class-plus-attributes trees), and the arrays inside them landed in the store's shared
# ``data/`` directory - once each, however many records or fields reference them.
#
# Comparison follows the same rule: an object is compared attribute by attribute, so a
# diff points at ``$.output.origin.x`` rather than telling you two ``Experiment`` objects
# are unequal. Objects are compared by their attributes, not by ``__eq__``, so a class with
# no ``__eq__`` still diffs usefully - and one whose ``__eq__`` is loose still gets checked
# strictly. A named tuple's diff names the field: ``$.output.ask``, not ``$.output[1]``.

from regression_records import compare  # noqa: E402

show = compare(
    Experiment("run-42", Point(0.0, 0.0), []),
    Experiment("run-42", Point(0.0, 0.5), []),
    root="$.output",
)
print(show.summary())
print(compare(Quote(99.5, 100.5), Quote(99.5, 101.0), root="$.output").summary())
