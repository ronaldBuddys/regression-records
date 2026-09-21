"""
Custom classes : recording objects by their state
=================================================

``to_json_tree()`` and ``from_json_tree()`` are the store/load pair the decorators are built
on, and they are usable directly - the successor to the old ``json_store`` /
``load_json_store``.

The headline: **an ordinary class needs no registration.** Anything exposing dict-shaped
attribute state - a ``__dict__``, or a ``__getstate__()`` returning a dict - is recorded
attribute by attribute by the generic object serializer, which sits at the bottom of the
registry as the fallback. You register a serializer only for types that have *no* such
state (example 9).
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
from _shared.models import Experiment, Point, Reading

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
# its ``__dict__``.

print(json.dumps(to_json_tree(Point(1.0, 2.5), writing), indent=2))

# %%
# A class that chooses its own state
# ----------------------------------
#
# ``Reading`` holds a live file handle, which has no business in a record. Its
# ``__getstate__`` names the two attributes that actually describe it, and ``__setstate__``
# puts the handle back as ``None`` on the way in. This is the same protocol ``pickle`` uses,
# so a class that already round-trips through pickle needs nothing new.

sample = Reading("thermocouple-3", np.array([20.1, 20.4, 20.2]))
print(sample)
print(json.dumps(to_json_tree(sample, writing), indent=2))

# %%
# The array became a reference to a ``.npy`` file rather than a list of numbers: full
# precision, full dtype, and content-addressed, so two equal arrays anywhere in the tree
# share one file.
#
# Nesting
# -------
#
# Serialization recurses, so objects inside objects inside lists all work. This is one
# ``Experiment`` holding a ``Point`` and two ``Reading`` objects.

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
print("__setstate__ ran:   ", loaded.readings[0].handle is None)

# %%
# What actually happens on load
# -----------------------------
#
# The class is re-imported from the ``module`` / ``qualname`` in the record, then rebuilt with
# ``object.__new__(cls)`` followed by ``__setstate__(state)`` - or ``instance.__dict__.update``
# where there is no ``__setstate__``.
#
# ``__init__`` is never called. That is what makes a record loadable when a constructor
# needs a database, a file, or an argument nobody kept; it also means an invariant that
# only ``__init__`` establishes will not be re-established for you. A ``__setstate__`` is the
# place to put anything a rebuilt instance needs.

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
# **2. The state must be dict-shaped.** A ``__slots__`` class has no ``__dict__``, and a
# ``__getstate__`` may return a tuple - neither can be recorded attribute by attribute, and
# both are told so, pointing at the custom-``Serializer`` route in example 9.

from _shared.models import Vec3  # noqa: E402 - kept beside its use

try:
    to_json_tree(Vec3(1.0, 2.0, 3.0), writing)
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
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
# class-plus-state trees), and the arrays inside them landed in the store's shared ``data/``
# directory - once each, however many records or fields reference them.
#
# Comparison follows the same rule: a custom object is compared attribute by attribute,
# so a diff points at ``$.output.origin.x`` rather than telling you two ``Experiment`` objects are
# unequal. Objects are compared by *state*, not by ``__eq__``, so a class with no ``__eq__``
# still diffs usefully - and one whose ``__eq__`` is loose still gets checked strictly.

from regression_records import compare  # noqa: E402

show = compare(
    Experiment("run-42", Point(0.0, 0.0), []),
    Experiment("run-42", Point(0.0, 0.5), []),
    root="$.output",
)
print(show.summary())
