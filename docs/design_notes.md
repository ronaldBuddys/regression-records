# Design notes

Decisions that are not visible from the API: options that were considered and rejected,
and things examined and deliberately left alone. They are written down so that they read
as chosen rather than overlooked, and are not re-raised as fresh findings. Decisions
about the on-disk format are in [Persistence](persistence.md); those about diffing are
in [Comparison](comparison.md).

## Recording and checking are separate

Until 0.2.0 one decorator did both: an unseen call was recorded, a seen one was
validated in place, and four policy enums (`RunMode`, `OnNewCase`, `OverwritePolicy`,
`OnMismatch`) chose between them on every decoration. Several of their combinations
passed without testing anything, accepting a change meant editing the decorator and
editing it back, and validation depended on a call's input hash matching a record's,
which a library upgrade could silently break. Now the decorator only records, and
`replay()` only checks.

- **The switch is an environment variable**, `REGRESSION_RECORDS`, read on every call.
  A module global would not reach subprocesses or CI jobs; reading it per call lets a
  test flip it. An unrecognised value raises rather than meaning "off", so a typo
  cannot silently stop recording.
- **Records are never replaced or deleted automatically.** If the new behavior is the
  bug, an overwrite would have lost the correct record. Deleting the old record is a
  deliberate, reviewable act.
- **The record id is derived from the content**, so that never replacing records does
  not mean accumulating copies: an unchanged recording lands on the filename it already
  has, across runs and parallel workers, with no locking.
- **Nothing is looked up by input hash.** Replay checks every record on its own. Two
  records of one call that disagree give a pass and a failure, which is accurate;
  deciding what a failure means is left to the user (and to a later tool for tracking
  results over time, which is why `ReplayReport.to_dict()` exists).
- **Replay calls the function beneath the decorator**, and forces recording off for
  the whole process while it runs, so replaying can never write a record.
- **Tolerance lives on replay.** Recording stores values exactly and has no use for it;
  the same records can be replayed under different tolerances.

## Replaying methods

- **The receiver is rebuilt without `__init__`**, from the whole recorded state, as
  unpickling does. A constructor may need a database, a file or an argument nobody
  kept, and the record already holds what the method acts on. The cost is that an
  invariant only `__init__` establishes is not re-established; a `factory=` is the
  place for it.
- **A partial state needs a factory, checked at decoration.** A record made with
  `state_attrs` or `exclude_attrs` holds some attributes of an object whose other
  attributes (a connection, say) the method may need. Rebuilding from it alone would
  replay against a half-built object and report its failures as regressions, so the
  decorator refuses to be applied without `factory=`, rather than replay discovering
  the gap on every record.
- **The current decorator decides how to rebuild.** Records carry no state spec or
  factory, so the record format did not change and a factory - often a lambda, or
  something tied to the test environment - never has to be importable by name. The
  price is that changing `state_attrs` or `exclude_attrs` after recording changes what
  the state after is compared on, and old records need re-recording.
- **`state_before` is not compared.** It is the call's input, rebuilt into the
  receiver; `state_after` is compared with the return value, which is what catches a
  method that returns the right answer by way of the wrong mutation.

## Refused at decoration

A lambda, a closure, a `functools.partial` and anything defined in `__main__` are
refused when decorated. They used to be recorded and then replayed as
`not_importable`, which meant a store could fill with records no replay would ever
check, and the first sign was the first replay. `__main__` is refused even though a
replay from the same script would find it, because from pytest or CI it would not.

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
is different, and does raise: hiding it would record frames by their internal
attributes.)

## Performance, examined and left alone

- **The switch is read from the environment on every call.** It costs a dictionary
  lookup, and caching it would break a test that flips it.
- **`registry.find()` per node in the comparison walker** is a handful of `isinstance`
  checks, immaterial for record-sized trees, and short-circuiting it would break the
  extension point above.
- **Arguments are hashed from a buffer view**, not `tobytes()`, which dropped a full
  copy of every array argument from every call. A test pins that the digest is the same.

## The examples

- **They run in a subprocess each** in the test suite, each copied alone into an empty
  directory. Documentation that executes is the only kind that cannot go stale quietly;
  a subprocess rather than an import because each example defines module-level
  decorated functions and one registers serializers - state that would leak between
  them in one process; alone, because the gallery offers every page as a single-file
  download, which must run as it is.
- **Each page imports its code under test from `regression_records._examples`.** A class
  recorded by its state is re-imported by name on load, and replay imports a recorded
  function by name, so neither can be defined in `__main__` - which a script or a
  notebook cell is. The pages used to import that code from a shared
  `examples/_shared/` package, which a downloaded notebook does not have; then each page
  wrote its own copy to a module in a temporary directory and imported it from there,
  which worked but buried the point of the page under setup. `_examples` ships with the
  installed package instead, so a page imports its code the way a project's script
  imports its own package. It is not public API, says so in its own docstring, and may
  change without notice. Module names are unique across pages, because the docs build
  runs every page in one process, where `sys.modules` would hand a later page an earlier
  page's module of the same name.
- **A page prints the code it calls**, with `print(inspect.getsource(...))`: the whole
  module when it is short, otherwise each function or class beside the section that
  first uses it. Standard output is the one form that survives all three renderings: a
  `literalinclude` stays raw reStructuredText in the downloaded notebook, and a
  highlighted `_repr_html_` lands under `.. only:: html`, so it is missing from the PDF.
- **Stores are fresh per run, not cleaned.** `example_store(name)` is
  `<root>/<name>`, where the root is a temporary directory made once per process, or
  `$REGRESSION_RECORDS_EXAMPLE_STORE` for a reader who wants to keep the records. A run
  never meets an earlier run's records, so no page deletes anything, and the library
  never deletes a directory a reader named. The cost: a second run into the same named
  root finds the first run's records and fails its counts, which the docstring says.
  A page that starts a second process passes it the root in that same variable, or the
  child would record into a temporary directory of its own.
- **Writing them found real gaps**, fixed rather than documented around: there was no
  way to get "the default registry plus one entry" until `copy()`, and a `path_overrides`
  entry naming a dataframe column had no effect until resolution moved into
  `compare_columns`, the one place every dataframe comparator's columns pass through.

## Objects are their attributes (0.4.0)

Until 0.3 an object was recorded through `__getstate__()`, and only when that returned a
dict. A `__slots__` class, a slotted dataclass, or anything whose state was not
dict-shaped needed a `Serializer` of its own, which did not scale across a codebase of
dataclasses, named tuples and slotted containers of arrays. Since 0.4.0:

- **An object is its attributes**: its `__dict__` plus every slot set anywhere in its
  MRO, read directly. One definition (`_attributes.py`) serves recording, comparison,
  method-state capture and receiver rebuild, so they cannot disagree.
- **`__getstate__` / `__setstate__` are ignored.** Honouring them would bring back
  per-class behaviour and couple records to that code as it was when recorded. The cost
  is that a class which used them to drop a cache or a handle no longer can; a handle
  refuses the value, and a cache is recorded and compared like any attribute.
- **Rebuilding is `object.__new__` plus the attributes set back**, a slot through its
  descriptor (so a frozen dataclass's `__setattr__` does not intervene) and anything else
  in `__dict__`. Methods come from the code as it is at load time, which is the point of
  replaying against new code. The accepted cost is coupling to attribute names, private
  ones included: a renamed attribute makes old records not load (a removed slot, an added
  dataclass or named-tuple field is reported as such) or show up as a named diff.
- **No registration.** Storing a class by accident is guarded against by refusing what
  cannot be rebuilt faithfully instead: a class in the MRO implemented in C that holds
  state of its own (`Py_TPFLAGS_IMMUTABLETYPE` with a larger layout or members, as for
  file handles and exceptions), or one `object.__new__` refuses (datetimes, locks,
  `mappingproxy`). An attribute that cannot be stored refuses the whole value, never
  silently drops out: a record that lost an attribute would replay as a pass.
- **Serializers are for leaf values only.** The generic object serializer and the
  registries' `fallback=` machinery are gone. The object serializer sits outside every
  registry's search order: the tree walker falls back to it when nothing claims a
  value, and each registry holds it only to load the reserved `"object"` /
  `"namedtuple"` tags, so loading has a single dispatch by tag. `mappingproxy` and dicts with non-string keys became built-in
  leaves.

## Why Python 3.11

`typing.Self` and `datetime.UTC` are 3.11 additions used throughout. Telling a C class
from a Python one relies on `Py_TPFLAGS_IMMUTABLETYPE`, which static types carry only
from 3.10.
