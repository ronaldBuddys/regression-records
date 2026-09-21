"""
SerializerRegistry : writing a custom serializer
================================================

Example 8 covered the types that need nothing. This one covers the rest:

* a class with no dict-shaped state - ``__slots__``, or a ``__getstate__`` returning a
  tuple - which cannot be recorded attribute by attribute;
* a class that *could* be recorded that way but shouldn't, because it has a compact
  binary form, or its attributes are noise;
* a type whose differences deserve a better description than "these two are unequal".

The first two want a ``Serializer``, the third a ``Comparator``. They are independent -
either can be written without the other.
"""

import sys
from pathlib import Path

# sphinx_gallery_start_ignore
# The examples import their shared helpers from the directory this file lives in.
sys.path.insert(
    0, str(Path(__file__).parent if "__file__" in globals() else Path.cwd())
)
# sphinx_gallery_end_ignore

import numpy as np
from _shared.demo import fresh_store, show_record, show_result
from _shared.models import Vec3, Waveform

from regression_records import (
    Diff,
    RegressionMismatchError,
    SerializationError,
    compare,
    record_func,
)
from regression_records.comparison import default_comparator_registry
from regression_records.serialization import (
    SerializationContext,
    default_registry,
    to_json_tree,
)
from regression_records.serialization.naming import content_filename

STORE = fresh_store("09_custom_serializer")

# %%
# The gap to fill
# ---------------
#
# ``Vec3`` uses ``__slots__``, so there is no ``__dict__`` to record and the generic object
# serializer refuses it rather than recording something misleading.

scratch = SerializationContext(data_dir=STORE / "scratch", registry=default_registry)

try:
    to_json_tree(Vec3(1.0, 2.0, 3.0), scratch)
except SerializationError as e:
    print(f"SerializationError: {e}")

# %%
# A serializer
# ------------
#
# Four members, no base class to inherit - a ``Serializer`` is a ``Protocol``:
#
# * ``type_tag`` - written into the JSON as ``__type__``, and how ``load`` is found again
#   later. Namespace it (``"myapp.Vec3"``); it is a permanent part of your record format.
# * ``can_handle(obj)`` - claim the type, and only the type.
# * ``store(obj, ctx)`` - return a JSON-safe dict. Do not include the type tag; that is
#   added for you.
# * ``load(blob, ctx)`` - rebuild the object from that dict.


class Vec3Serializer:
    type_tag = "examples.Vec3"

    def can_handle(self, obj):
        return isinstance(obj, Vec3)

    def store(self, obj, ctx):
        return {"xyz": list(obj.as_tuple())}

    def load(self, blob, ctx):
        return Vec3(*blob["xyz"])


# %%
# Registering it on a copy
# ------------------------
#
# ``default_registry`` is module-level state shared by everything in the process, so
# registering on it changes every decoration everywhere. Copy it, register on the copy,
# and inject that - the change is then scoped to the decorations that asked for it.
#
# A new serializer lands ahead of the generic object fallback but behind the built-in
# ones, so it can claim a type the fallback would otherwise take, without accidentally
# outranking the numpy or dataframe serializers. Pass ``priority=0`` if you really do need
# to outrank them.

registry = default_registry.copy()
registry.register(Vec3Serializer())


@record_func(STORE, serializer_registry=registry)
def reflect(v):
    return Vec3(-v.x, -v.y, -v.z)


print("returned:", reflect(Vec3(1.0, 2.0, 3.0)))
show_record(STORE, keys=["invocation", "output"])

# %%
# To register globally instead - appropriate in an application, less so in a library or a
# single test - use ``register_serializer(Vec3Serializer())``, which does the same thing to
# ``default_registry`` itself.
#
# A serializer that writes a side file
# ------------------------------------
#
# Anything large belongs beside the JSON rather than inside it. ``ctx`` gives you three
# things for that:
#
# * ``content_filename(prefix, *chunks, extension=...)`` - a content-addressed name, so
#   equal payloads collapse onto one file;
# * ``ctx.write_if_absent(filename, write_fn)`` - write it unless a file of that name is
#   already there (which, being content-addressed, means identical content);
# * ``ctx.ref(filename)`` - the reference string to store, prefixed so it resolves relative
#   to whichever directory will read it back.
#
# On the way in, ``ctx.resolve(path)`` turns that reference back into a real path.


class WaveformSerializer:
    type_tag = "examples.Waveform"

    def can_handle(self, obj):
        return isinstance(obj, Waveform)

    def store(self, obj, ctx):
        samples = np.ascontiguousarray(obj.samples)
        filename = content_filename(
            "wave",
            samples.view(np.uint8).reshape(-1).data,
            str(obj.rate).encode(),
            extension="npy",
        )
        ctx.write_if_absent(
            filename, lambda path: np.save(path, samples, allow_pickle=False)
        )
        return {"path": ctx.ref(filename), "rate": obj.rate, "n": int(samples.size)}

    def load(self, blob, ctx):
        return Waveform(np.load(ctx.resolve(blob["path"])), blob["rate"])


registry.register(WaveformSerializer())


@record_func(STORE, serializer_registry=registry)
def decimate(wave, factor):
    return Waveform(wave.samples[::factor], wave.rate // factor)


wave = Waveform(np.sin(np.linspace(0, 10, 1000)), rate=44100)
print("returned:", decimate(wave, 4))
show_record(STORE, keys=["output"], select=lambda r: "Waveform" in str(r.get("output")))

# %%
# Note what the record holds: a reference and two scalars, with the samples in a ``.npy``
# file next door. Keep ``store``'s payload small and descriptive - it is what a human reads
# in a ``git diff`` when the record changes.

print("side files:", sorted(p.name for p in STORE.rglob("wave-*.npy")))

# %%
# A comparator
# ------------
#
# Without one, a custom type falls to the generic object comparator, which compares
# attribute by attribute - already useful, and for ``Vec3`` it cannot even do that, since
# there are no attributes to walk:

show_result(compare(Vec3(1.0, 2.0, 3.0), Vec3(1.0, 2.0, 3.5)))

# %%
# A ``Comparator`` is another ``Protocol``, with one important difference from a serializer:
# it is dispatched when **either** side matches, so it must handle the case where only one
# of them is your type.
#
# It appends ``Diff`` objects to the list it is given rather than returning them - which is what
# lets a comparator both report its own findings and recurse into ``compare_fn`` for nested
# values without double-counting.


class Vec3Comparator:
    def can_handle(self, expected, actual):
        return isinstance(expected, Vec3) or isinstance(actual, Vec3)

    def compare(self, expected, actual, path, tolerance, compare_fn, diffs):
        if not (isinstance(expected, Vec3) and isinstance(actual, Vec3)):
            diffs.append(
                Diff(
                    path=path,
                    kind="type_mismatch",
                    expected=type(expected).__name__,
                    actual=type(actual).__name__,
                    message="only one side is a Vec3",
                )
            )
            return
        # Delegating each component back to compare_fn gets tolerance, NaN handling and
        # per-component paths for free, rather than reimplementing them here.
        for axis, e, a in zip(
            "xyz", expected.as_tuple(), actual.as_tuple(), strict=True
        ):
            compare_fn(e, a, f"{path}.{axis}")


# %%

comparators = default_comparator_registry.copy()
comparators.register(Vec3Comparator())

show_result(compare(Vec3(1.0, 2.0, 3.0), Vec3(1.0, 2.0, 3.5), registry=comparators))

# %%
# ``$.z`` instead of a bare ``$``, and the tolerance that judged it. Injected onto a
# decorator as ``comparator_registry=``, that is what a mismatch report looks like:

FLIP = -1.0


@record_func(STORE, serializer_registry=registry, comparator_registry=comparators)
def scale(v, k):
    return Vec3(v.x * k, v.y * k, v.z * k * FLIP)


scale(Vec3(1.0, 2.0, 3.0), 2.0)

FLIP = -1.001
try:
    scale(Vec3(1.0, 2.0, 3.0), 2.0)
except RegressionMismatchError as e:
    show_result(e.result)

# %%
# A checklist
# -----------
#
# * ``type_tag`` is permanent. Records written with one tag are read back by looking it up;
#   renaming it strands every record that used it.
# * ``store`` must return JSON-safe values (or nested values the registry can handle) -
#   ``load(store(x))`` should give back ``x``.
# * Content-address anything you write, so re-recording the same value does not multiply
#   files.
# * A comparator is dispatched if *either* side matches; check both.
# * Recurse through ``compare_fn`` for nested values rather than comparing them yourself,
#   or you will silently drop tolerance handling.
