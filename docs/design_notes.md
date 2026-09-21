# Design notes

Decisions that are not visible from the API: options that were considered and rejected,
and things examined and deliberately left alone. They are written down so that they read
as chosen rather than overlooked, and are not re-raised as fresh findings. Decisions
about the on-disk format are in [Persistence](persistence.md); those about diffing are
in [Comparison](comparison.md).

## Rejected simplifications

Both have their reasoning as a comment at the site.

- **Exact-type tests for primitives in `to_json_tree`**, which would delete the
  `np.generic` special case. Rejected: it would also reroute every other builtin
  subclass - `IntEnum`, `StrEnum`, anything subclassing `int` for units - to the generic
  object serializer, where an `Enum` member fails to record at all. Recording those as
  their underlying builtin is right, and `isinstance` is what says so.
- **Dropping `_EXCLUDED` in `comparison/objects.py`** as a restatement of registry
  order. Rejected: the premise is false. The walker consults the registry *before* its
  own structural branches, deliberately, so that a user-registered comparator for a dict
  or list subclass wins. That makes the exclusions load-bearing, and flipping the
  dispatch order would break that extension point to delete a few list entries.

## Not exported, on purpose

`compare_exceptions` takes an exception *type name* as a string - the shape the engine
wants, and a strange one to hand a user. The record model (`Invocation`,
`SerializedValue`, `ExceptionRecord`, `SchemaInfo`) and `comparison.paths` are
importable but not in `__all__`: stable enough to inspect a loaded record with, not
promised.

## No error for a missing extra

A `pl.DataFrame` passed on an install without polars cannot happen - without polars you
cannot construct one. So there is no "install the polars extra" error to raise. The
guarantee that *is* checked is that a bare install imports and records numpy end to end,
with the optional serializers simply absent. (A library that is installed but *broken*
is different, and does raise: hiding it would record frames through the generic object
fallback.)

## Performance, examined and left alone

- **Re-reading a record's JSON and side files on every validated call.** A cache would
  need invalidation, and consecutive calls generally hit different records anyway.
- **The double directory glob in `find_match`** happens only on the new-case path, which
  then writes a file regardless.
- **`registry.find()` per node in the comparison walker** is a handful of `isinstance`
  checks, immaterial for record-sized trees, and short-circuiting it would break the
  extension point above.
- **Arguments are hashed from a buffer view**, not `tobytes()`, which dropped a full
  copy of every array argument from every call. A test pins that the digest is the same.

## The examples

- **They run in a subprocess each** in the test suite. Documentation that executes is
  the only kind that cannot go stale quietly; a subprocess rather than an import because
  each example defines module-level decorated functions and one registers serializers -
  state that would leak between them in one process.
- **Sample classes live in `examples/_shared/models.py`.** A class recorded by its state
  is re-imported by name on load, so one defined in `__main__` - which a script or a
  notebook cell is - cannot be recorded at all. The examples would otherwise demonstrate
  the one thing that does not work.
- **Writing them found real gaps**, fixed rather than documented around: there was no
  way to get "the default registry plus one entry" until `copy()`, and a `path_overrides`
  entry naming a dataframe column had no effect until resolution moved into
  `compare_columns`, the one place every dataframe comparator's columns pass through.

## Why Python 3.11

Recording an object by its state calls `obj.__getstate__()`, which every object inherits
from `object` only from 3.11. The default answers `None` for an object whose `__dict__`
is empty, which is what keeps functions and lambdas from being "recorded" as an empty
state. On 3.10 that fell through to `__dict__`, so stores written there disagreed with
stores written anywhere else.
