"""
Serializer : adding a leaf type
===============================

Example 8 showed that a class of your own needs nothing: it is recorded by its
attributes. A ``Serializer`` is for the values that have no attributes to record - *leaf*
values. The built-in ones cover numpy arrays and scalars, polars and pandas frames,
datetimes, ``Decimal``, paths, bytes and the like. What is left is mostly C types from
compiled extensions, whose state lives below Python attributes.

This example adds one, stdlib ``array.array``, in two parts that are independent of each
other: a ``Serializer`` to record it, and a ``Comparator`` to describe how two of them
differ.
"""

import inspect
import sys
from array import array
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

from _shared import signals
from _shared.demo import fresh_store, show_cases, show_record, show_result

from regression_records import (
    Diff,
    SerializationError,
    compare,
    enable,
    replay,
)
from regression_records.comparison import default_comparator_registry
from regression_records.serialization import (
    SerializationContext,
    default_registry,
    to_json_tree,
)
from regression_records.serialization.naming import content_filename

STORE = fresh_store(signals.STORE.root.name)

# %%
# Recording is off unless switched on; this example switches it on for the whole
# script.

enable()

# %%
# The gap to fill
# ---------------
#
# An ``array.array`` is a C type: its samples are held in a C buffer, not in attributes,
# so there is nothing to record it by and it is refused rather than recorded as
# something misleading.

scratch = SerializationContext(data_dir=STORE / "scratch", registry=default_registry)

try:
    to_json_tree(array("d", [1.0, 2.0, 3.0]), scratch)
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
# A serializer
# ------------
#
# Four members, no base class to inherit - a ``Serializer`` is a ``Protocol``:
#
# * ``type_tag`` - written into the JSON as ``__type__``, and how ``load`` is found again
#   later. Namespace it (``"myapp.array"``); it is a permanent part of your record format.
#   ``"object"`` and ``"namedtuple"`` are reserved for objects stored by attributes.
# * ``can_handle(obj)`` - claim the type, and only the type.
# * ``store(obj, ctx)`` - return a JSON-safe dict. Do not include the type tag; that is
#   added for you.
# * ``load(blob, ctx)`` - rebuild the value from that dict.
#
# Anything large belongs beside the JSON rather than inside it, and ``ctx`` gives you
# three things for that:
#
# * ``content_filename(prefix, *chunks, extension=...)`` - a content-addressed name, so
#   equal payloads collapse onto one file;
# * ``ctx.write_if_absent(filename, write_fn)`` - write it unless a file of that name is
#   already there (which, being content-addressed, means identical content);
# * ``ctx.ref(filename)`` - the reference string to store, prefixed so it resolves relative
#   to whichever directory will read it back.
#
# On the way in, ``ctx.resolve(path)`` turns that reference back into a real path.


class ArraySerializer:
    type_tag = "examples.array"

    def can_handle(self, obj):
        return isinstance(obj, array)

    def store(self, obj, ctx):
        data = obj.tobytes()
        filename = content_filename(
            "samples", obj.typecode.encode(), data, extension="bin"
        )
        ctx.write_if_absent(filename, lambda path: path.write_bytes(data))
        return {"path": ctx.ref(filename), "typecode": obj.typecode, "n": len(obj)}

    def load(self, blob, ctx):
        samples = array(blob["typecode"])
        samples.frombytes(ctx.resolve(blob["path"]).read_bytes())
        return samples


# %%
# Registering it on a copy
# ------------------------
#
# ``default_registry`` is module-level state shared by everything in the process, so
# registering on it changes every decoration everywhere. Copy it, register on the copy,
# and inject that - the change is then scoped to the decorations that asked for it.
#
# A new serializer lands behind the built-in ones, so it cannot accidentally outrank the
# numpy or dataframe serializers. It is consulted before an object is stored by its
# attributes, so it could also take over a class of your own - rarely what you want,
# since that class then needs this serializer to load. Pass ``priority=0`` if you really
# do need to outrank the built-ins.
#
# The functions this example records live in ``examples/_shared/signals.py``, where
# replay can import them. That module builds its store on such a copy:
#
# .. code-block:: python
#
#    REGISTRY = default_registry.copy()
#    STORE = RecordStore(
#        EXAMPLE_STORES / "09_custom_serializer", serializer_registry=REGISTRY
#    )
#
#
# and the serializer is registered on that copy here, so that its code stays on this
# page. In your own code, register it in the module that records, so that any process
# importing the module can load its records.

registry = signals.REGISTRY
registry.register(ArraySerializer())

print(inspect.getsource(signals.decimate))
print("returned:", signals.decimate(array("d", [0.0, 0.5, 1.0, 1.5, 2.0]), 2))
show_record(STORE, keys=["invocation", "output"])

# %%
# Note what the record holds: a reference, the type code and a length, with the samples
# in a ``.bin`` file next door. Keep ``store``'s payload small and descriptive - it is
# what a human reads in a ``git diff`` when the record changes.
#
# To register globally instead - appropriate in an application, less so in a library or
# a single test - use ``register_serializer(ArraySerializer())``, which does the same
# thing to ``default_registry`` itself.

print("side files:", sorted(p.name for p in STORE.rglob("samples-*.bin")))

# %%
# A comparator
# ------------
#
# Without one, a leaf value with no attributes to walk is compared by ``==``: correct,
# but all it can say is that two arrays differ, with no tolerance and no position:

show_result(compare(array("d", [1.0, 2.0, 3.0]), array("d", [1.0, 2.0, 3.5])))

# %%
# A ``Comparator`` is another ``Protocol``, with one important difference from a serializer:
# it is dispatched when **either** side matches, so it must handle the case where only one
# of them is your type.
#
# It appends ``Diff`` objects to the list it is given rather than returning them - which is what
# lets a comparator both report its own findings and recurse into ``compare_fn`` for nested
# values without double-counting.


class ArrayComparator:
    def can_handle(self, expected, actual):
        return isinstance(expected, array) or isinstance(actual, array)

    def compare(self, expected, actual, path, tolerance, compare_fn, diffs):
        if not (isinstance(expected, array) and isinstance(actual, array)):
            diffs.append(
                Diff(
                    path=path,
                    kind="type_mismatch",
                    expected=type(expected).__name__,
                    actual=type(actual).__name__,
                    message="only one side is an array.array",
                )
            )
            return
        # Handing the samples back to compare_fn as lists gets the length check,
        # tolerance, NaN handling and per-sample paths for free, rather than
        # reimplementing them here.
        compare_fn(expected.tolist(), actual.tolist(), path)


# %%

comparators = default_comparator_registry.copy()
comparators.register(ArrayComparator())

show_result(
    compare(
        array("d", [1.0, 2.0, 3.0]),
        array("d", [1.0, 2.0, 3.5]),
        registry=comparators,
    )
)

# %%
# ``$[2]`` instead of a bare ``$``, and the tolerance that judged it.
#
# Replaying with both
# -------------------
#
# Replay needs the serializer to load the records, and the comparator to report their
# differences this way. One more recorded function, ``amplify``, gives the replay
# something to catch:

print(inspect.getsource(signals.amplify))
print("returned:", signals.amplify(array("d", [1.0, 2.0, 3.0]), 2.0))

# %%
# Pass replay the ``RecordStore`` rather than its path, so that it loads with the
# store's registry, and the comparators as ``comparator_registry=``. ``GAIN`` is
# changed for the replay, standing in for an edit to ``signals.py``: ``amplify``
# differs, and ``decimate`` still passes.

signals.GAIN = 1.001
try:
    report = replay(signals.STORE, comparator_registry=comparators)
finally:
    signals.GAIN = 1.0

show_cases(report)
show_result(report.failures[0].comparison)

# %%
# A checklist
# -----------
#
# * Write a serializer only for a leaf value. A class of your own is recorded by its
#   attributes, and a serializer for it is one more thing its records need to load.
# * ``type_tag`` is permanent. Records written with one tag are read back by looking it up;
#   renaming it strands every record that used it.
# * ``store`` must return JSON-safe values (or nested values the registry can handle) -
#   ``load(store(x))`` should give back ``x``.
# * Content-address anything you write, so re-recording the same value does not multiply
#   files.
# * A comparator is dispatched if *either* side matches; check both.
# * Recurse through ``compare_fn`` for nested values rather than comparing them yourself,
#   or you will silently drop tolerance handling.
