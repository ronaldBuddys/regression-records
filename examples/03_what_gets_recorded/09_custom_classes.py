"""
``to_json_tree``, ``from_json_tree`` : custom classes stored by their attributes
================================================================================

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
attributes to record (:ref:`sphx_glr_auto_examples_03_what_gets_recorded_10_custom_serializer.py`).
"""

# %%
# Setup: the code under test
# --------------------------
#
# A recorded object is stored with its class's module and qualified name, and loading it
# re-imports the class by that name - so the classes have to live in an importable
# module (rule 1 below shows what happens otherwise). In your project that is your own
# package; here the page writes two modules to a fresh temporary directory: ``models.py``
# with the sample classes, and ``experiments.py`` with a recorded function that takes
# and returns them, its store beside it.
#
# Between them the classes cover the shapes a class can take: ``Point`` has plain
# attributes, ``Reading`` and ``Experiment`` nest with arrays inside, ``Quote`` is a
# named tuple, ``Vec3`` has ``__slots__`` and no ``__dict__``, and ``Calibration`` is a
# frozen, slotted dataclass holding a read-only mapping. ``LiveReading`` holds an open
# file, which is not data.

import json
import sys
import tempfile
from pathlib import Path

WORKDIR = Path(tempfile.mkdtemp(prefix="regression-records-"))
(WORKDIR / "models.py").write_text('''
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np


class Quote(NamedTuple):
    """A named tuple. Recorded with its class and its fields by name."""

    bid: float
    ask: float


class Point:
    """Plain attributes. Recorded straight out of ``__dict__``."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f"Point(x={self.x!r}, y={self.y!r})"


class Reading:
    """A named series of samples."""

    def __init__(self, sensor: str, samples: np.ndarray):
        self.sensor = sensor
        self.samples = samples

    def __repr__(self):
        return f"Reading(sensor={self.sensor!r}, samples={self.samples!r})"


class LiveReading(Reading):
    """A reading that also holds a live file handle, which has no place in a record."""

    def __init__(self, sensor: str, samples: np.ndarray):
        super().__init__(sensor, samples)
        self.handle = open(os.devnull)


class Experiment:
    """Nests a ``Point`` and ``Reading`` objects - recorded recursively."""

    def __init__(self, name: str, origin: Point, readings: list[Reading]):
        self.name = name
        self.origin = origin
        self.readings = readings

    def __repr__(self):
        return (
            f"Experiment(name={self.name!r}, origin={self.origin!r}, "
            f"readings={self.readings!r})"
        )


class Vec3:
    """``__slots__``, so no ``__dict__``: each slot is recorded as an attribute."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vec3({self.x!r}, {self.y!r}, {self.z!r})"


@dataclass(frozen=True, slots=True)
class Calibration:
    """Frozen and slotted, with a read-only mapping of per-sensor offsets."""

    instrument: str
    offsets: Mapping[str, float]

    def __post_init__(self):
        object.__setattr__(self, "offsets", MappingProxyType(dict(self.offsets)))
''')
(WORKDIR / "experiments.py").write_text('''
from pathlib import Path

from models import Experiment, Point

from regression_records import record_func

STORE = Path(__file__).parent / "store"


@record_func(STORE)
def recentre(experiment: Experiment, origin: Point) -> Experiment:
    """Return a copy of ``experiment`` moved to ``origin``."""
    return Experiment(name=experiment.name, origin=origin, readings=experiment.readings)
''')
sys.path.insert(0, str(WORKDIR))

import experiments
import numpy as np
from models import Calibration, Experiment, Point, Quote, Reading, Vec3

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

ROOT = WORKDIR / "trees"
print("trees:", ROOT)

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

point_tree = to_json_tree(Point(1.0, 2.5), writing)
print(json.dumps(point_tree, indent=2))
assert point_tree["class"] == {"module": "models", "qualname": "Point"}

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
    assert type(loaded) is type(value)

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
# share one file. Drawn as a tree, the record is the class at the root, its attributes
# as branches, and a side file at each array:

import matplotlib.pyplot as plt

KIND_COLORS = {"object": "#2a78d6", "array": "#eb6834", "value": "#52514e"}


def layout(node, key="", depth=0, parent=None, nodes=None):
    """Flatten a stored tree into nodes: ``(depth, row, label, kind, parent row)``."""
    nodes = [] if nodes is None else nodes
    row = len(nodes)
    if isinstance(node, dict) and node.get("__type__") == "object":
        cls = node["class"]
        label, kind = f"{key}{cls['qualname']}  ({cls['module']})", "object"
        children = list(node["attrs"].items())
    elif isinstance(node, dict) and node.get("__type__") == "numpy.ndarray":
        name = Path(node["path"]).name
        label = f"{key}{name[:12]}...npy  {node['dtype']} {tuple(node['shape'])}"
        kind, children = "array", []
    elif isinstance(node, list):
        label, kind = f"{key}list[{len(node)}]", "value"
        children = [(f"[{i}]", child) for i, child in enumerate(node)]
    else:
        label, kind, children = f"{key}{node!r}", "value", []
    nodes.append((depth, row, label, kind, parent))
    for name, child in children:
        layout(child, f"{name} = ", depth + 1, row, nodes)
    return nodes


nodes = layout(tree)
assert sum(kind == "array" for *_, kind, _ in nodes) == 2
assert sum(kind == "object" for *_, kind, _ in nodes) == 4

fig, ax = plt.subplots(figsize=(7.5, 4.2))
depth_of = {row: depth for depth, row, *_ in nodes}
for depth, row, label, kind, parent in nodes:
    if parent is not None:
        ax.plot(
            [depth_of[parent] + 0.15, depth_of[parent] + 0.15, depth],
            [parent + 0.3, row, row],
            color="#c3c2b7",
            lw=1,
        )
    ax.text(
        depth + 0.05,
        row,
        label,
        va="center",
        fontsize=9,
        family="monospace",
        color=KIND_COLORS[kind],
        fontweight="bold" if kind == "object" else "normal",
    )
LEGEND = {
    "object": "object: class + attributes",
    "array": "array: a .npy side file",
    "value": "plain JSON value",
}
for kind, color in KIND_COLORS.items():
    ax.plot([], [], "s", color=color, label=LEGEND[kind])
ax.set_xlim(-0.1, 6.2)
ax.set_ylim(len(nodes) - 0.4, -0.8)
ax.axis("off")
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("One recorded Experiment, as stored", loc="left")
fig.tight_layout()

# %%
# Load it back
# ------------
#
# Through a real file, so nothing survives in memory between the two halves.

loaded = from_json_tree(json.loads((ROOT / "experiment.json").read_text()), reading)

print(loaded)
assert type(loaded) is Experiment
assert type(loaded.origin) is Point and type(loaded.readings[0]) is Reading
assert loaded.origin == experiment.origin, "the nested value did not round-trip"
np.testing.assert_array_equal(loaded.readings[0].samples, [20.1, 20.4, 20.2])
assert loaded.readings[0].samples.dtype == np.float64

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
# notebook cell, like this page - cannot be re-imported from a fresh process, and is
# refused at store time rather than producing a record nobody can read:

from regression_records import SerializationError


class DefinedRightHere:
    def __init__(self):
        self.x = 1


try:
    to_json_tree(DefinedRightHere(), writing)
except SerializationError as e:
    print(f"SerializationError: {e}")
else:
    raise AssertionError("a class in __main__ is refused")

# %%
# This is why this page writes its classes to ``models.py`` rather than defining them in
# a cell. Move a class into any importable module and it records fine.
#
# **2. Every attribute must be data.** An attribute is recorded by the same rules as any
# other value, so one that is not data - an open file, a lock, a function, an object
# from a compiled extension - refuses the whole object. It is never skipped: a record
# that silently lost an attribute would replay as a pass. ``LiveReading`` holds an open
# file:

from models import LiveReading

try:
    to_json_tree(LiveReading("thermocouple-3", np.array([20.1])), writing)
except SerializationError as e:
    print(f"SerializationError: {e}")
else:
    raise AssertionError("an open file is not data")

# %%
# Record the data such an object holds instead - here, a plain ``Reading``. For a method
# whose receiver holds one, ``record_method(exclude_attrs=...)`` leaves it out of the
# recorded state (:ref:`sphx_glr_auto_examples_03_what_gets_recorded_07_recording_methods.py`).
#
# The same thing, through a decorator
# -----------------------------------
#
# Everything above is what happens inside ``record_func`` and ``record_method``. Custom
# objects work as arguments, as return values, and as recorded method state, with no
# configuration at all.

from regression_records import enable

STORE = experiments.STORE
print("store:", STORE)

# Recording is off unless switched on; this page switches it on from here to the end.
enable()

experiments.recentre(experiment, Point(5.0, 5.0))
record_text = next(STORE.rglob("rec-*.json")).read_text()
print(json.dumps({"output": json.loads(record_text)["output"]}, indent=2))

references = record_text.count(".npy")
on_disk = len(list(STORE.rglob("*.npy")))
print("\n.npy references in the record:", references)
print(".npy files on disk:           ", on_disk)
assert (references, on_disk) == (4, 2)

# %%
# The arguments are stored the same way (the record's ``invocation.args`` holds the same
# class-plus-attributes trees), and the arrays inside them landed in the store's shared
# ``data/`` directory - once each, however many records or fields reference them.
#
# Comparison follows the same rule: an object is compared attribute by attribute, so a
# diff points at ``$.output.origin.y`` rather than telling you two ``Experiment`` objects
# are unequal. Objects are compared by their attributes, not by ``__eq__``, so a class with
# no ``__eq__`` still diffs usefully - and one whose ``__eq__`` is loose still gets checked
# strictly. A named tuple's diff names the field: ``$.output.ask``, not ``$.output[1]``.

from regression_records import compare

moved = compare(
    Experiment("run-42", Point(0.0, 0.0), []),
    Experiment("run-42", Point(0.0, 0.5), []),
    root="$.output",
)
print(moved.summary())
quoted = compare(Quote(99.5, 100.5), Quote(99.5, 101.0), root="$.output")
print(quoted.summary())
assert [d.path for d in moved.diffs] == ["$.output.origin.y"]
assert [d.path for d in quoted.diffs] == ["$.output.ask"]

# %%
# **Next:** :ref:`sphx_glr_auto_examples_03_what_gets_recorded_10_custom_serializer.py` -
# a serializer for a leaf value with no attributes to record.
