# Persistence

A record store is a directory of JSON files and the data files they reference. It is
meant to be read by people, diffed by `git`, and committed alongside the code it
describes. Nothing in it is pickled by default, and nothing in it is a database.

## Layout

```
tests/regression_data/
  myapp/
    pricing/                                  # module path, as directories
      compute_score/                          # the function's qualname
        records/
          rec-3f9a1c2b7e6d4a10-0f3a9c2e.json
          rec-9c1a7d55e2b1f004-7b1d44aa.json
        data/
          arr-3f9a1c2b7e6d4a10.npy
          df-9c1a7d55e2b1f004.parquet
      Portfolio.rebalance/                    # qualname encodes the class
        records/
          rec-11a2b3c4d5e6f708-99aa11bb.json
        data/
          df-a1b2c3d4e5f60718.parquet
```

Each function or method gets one directory, holding a `records/` subdirectory of
invocations and a `data/` subdirectory of the values too large or too structured to sit
inside JSON. Replay walks every `records/` directory under the root.

## Identity: which directory a call belongs to

A record's identity is `(module, qualname)`, taken from the *undecorated* function at
decoration time.

`__qualname__` already encodes class nesting - `Portfolio.rebalance`, `Outer.Inner.run` -
which is what keeps two same-named methods on different classes from colliding, a
distinction a bare method name loses. It is also how replay finds a method's class
again: the qualname without its last part.

Because replay imports a function by this identity, a callable without an importable
one is refused when decorated (`TypeError`): a lambda (`<lambda>`), a function or class
defined inside a function (`<locals>`), a `functools.partial` (no qualname at all), and
anything whose module is `__main__`, which is a different module in the process that
replays. Stores written by earlier versions may hold records under such identities;
they load, and replay reports them as `not_importable`.

Each record also stores a **`code_hash`**: a hash of the function's source, normalized to
ignore whitespace and comments (falling back to the compiled bytecode when source is
unavailable, and to the literal string `"unknown"` when even that is). It is
**informational only** - never part of the directory path or of the record id. That
is deliberate: a refactor must not orphan a function's history, and re-recording a
refactored function that behaves as before writes nothing. What it buys you is the drift
flag - `ReplayCase.code_drift`, and `CodeDriftWarning` under `replay_tests` - which tells
you whether a pass means "behavior held across a real change" or merely "nothing
changed".

`signature` is stored for the same reason - human debugging - and is never compared.

### Moving a function

Not auto-resolved, because it is genuinely ambiguous: a function that vanished from one
module and a similar one that appeared in another are not necessarily the same function.
Until it is resolved, replay cannot import the recorded name and reports its records as
`not_importable`. `RecordStore.rename_identity(old, new)` moves them by hand:

```python
store.rename_identity(
    FunctionIdentity.from_function(old_module.compute_score),
    FunctionIdentity.from_function(new_module.compute_score),
)
```

It moves the directory and updates each record's stored `module`/`qualname` so the files
and the path they live at agree. `code_hash` and `signature` are left as recorded - they
describe the code each record was *captured from*, and rewriting them would silently
erase the drift signal you would otherwise get if the implementation had also changed.

## The input hash: grouping records of one call

Each record carries an **`input_hash`**: a sha256 over the canonicalized *serialized*
arguments - `json.dumps(..., sort_keys=True, separators=(",", ":"))` of `{"args": ...,
"kwargs": ...}` (`regression_records.records.compute_input_hash`).

It leads the record's filename, so the records of one call sort together, and a
`ReplayCase` carries it so a report can be grouped by call. **Nothing is looked up by
it.** Replay checks every record on its own, so a hash that changes - after a library
upgrade that serializes an argument differently, say - costs an extra record on the
next recording, not a missed check.

Serializing first is what makes the hash meaningful. By that point every external value
is already content-addressed, so two equal numpy arrays contribute identically to the
hash regardless of object identity, in this process or any other.

Arguments are normalized through `inspect.signature().bind()` plus `apply_defaults()`
before hashing, so `f(2, 3)`, `f(2, b=3)`, `f(a=2, b=3)` and `f(2)` with `b` defaulting to
`3` are all **one** call rather than four.

For methods, `self` is excluded from the recorded arguments and from the hash; the
receiver is described by the captured state instead. `args` is therefore always `[]` for
a method: once the receiver is gone, positional order no longer describes the call, and
`apply_defaults()` has already named everything.

**Arguments are never compared with tolerance.** Tolerances apply to a call's output and
state on replay: two invocations with slightly different arguments are two calls, each
recorded and replayed on its own.

The function's identity is deliberately *not* mixed into the hash - records are already
namespaced by identity on disk, and including it would change every hash whenever a
function moved module.

## Records are never replaced

A record's id is derived from its content (`regression_records.records.content_record_id`):
the identity, the input hash, the serialized output or the exception's *type*, and for a
method the state before and after. What replay does not compare stays out - the time,
the code hash, the exception message and traceback, and tags.

So recording the same behavior again lands on the same filename, and nothing is
written: not by a second run, and not by parallel pytest-xdist workers, which converge
on one file. Recording *changed* behavior gets a new id and a new file beside the old
one. The store never deletes or replaces a record itself; removing the old one is a
manual act, followed by `prune` for the data files it leaves.

Within one process, a call is recorded once: repeating it with the same arguments (and,
for a method, the same starting state) is not serialized and written again.

Records written before version 0.2.0 have random ids. They replay like any other; a
re-recording of the same behavior simply gets a content id and a second file.

## File naming

**Records** are `rec-<input_hash[:16]>-<record_id[:8]>.json`. The input hash leads so
that the records of one call sort together; the full hash is inside the file. The
record id follows so that distinct recordings of the same inputs - before and after an
intended change - sit side by side. Because the id is derived from the content, the
filename of a recording is known before it is written, and an existing file with that
name already holds the same behavior.

**Data files** are `<prefix>-<sha256[:32]>.<ext>`, hashed over the file's own content -
except arrays, which are hashed over their elements, shape and dtype. Each of those
chunks is prefixed with its length before hashing, so they cannot run into each other.
Files written by earlier versions have a 16-character hash of the chunks simply
concatenated. Records reference files by path, so those names keep resolving.

| Prefix | Type | Format |
|---|---|---|
| `arr` | `numpy.ndarray` | `.npy`, via `np.save(..., allow_pickle=False)` |
| `df` | `polars.DataFrame` | `.parquet` |
| `series` | `polars.Series` | `.parquet`, stored as a one-column frame |
| `pddf` | `pandas.DataFrame` | `.parquet`, or `.pkl` on fallback |
| `pdseries` | `pandas.Series` | as above, stored as a one-column frame |

Because the name *is* the content hash, a file that already exists is known to hold
identical content and is not rewritten. A value recorded before and after the longer
hash is stored twice, once under each name. Deduplication is scoped per identity directory,
not globally: one function's history stays a self-contained unit you can move, copy, or
delete without consulting the rest of the store.

Numpy scalars (`np.float64`, `np.datetime64`, ...) are stored **inline** in the JSON with
their dtype, not as side files - they are small, and being able to read them in the
record is worth more than uniformity.

So are the standard library types JSON has no form for, each tagged so it loads back as
its own type - which is also what lets a recorded *argument* be passed back to the
function it came from: `tuple`, `set`, `frozenset`, `bytes`, `bytearray`, `datetime`
(with a `ZoneInfo` zone and `fold`), `date`, `time`, `timedelta`, `Decimal` (as a
string, so no digit or exponent is lost), `complex`, `pathlib` paths, `Enum` members (by
class and member name), `types.MappingProxyType` (as the mapping it wraps), dicts with
non-string keys (as `[key, value]` pairs, each converted like any value) and non-finite
floats. With pandas installed, `Timestamp` and `Timedelta` keep their nanoseconds and
their type.

## Objects

Any other object is stored as its class plus its attributes - its `__dict__` and every
slot set anywhere in its class hierarchy - each converted by the same rules:

```json
{
  "__type__": "object",
  "class": { "module": "myapp.market", "qualname": "Quote" },
  "attrs": { "bid": 99.5, "ask": 100.5 }
}
```

That covers plain classes, `__slots__` classes, frozen and slotted dataclasses and named
tuples (stored by field name) alike, with no registration and no serializer. Loading
imports the class, allocates an instance with `object.__new__` and sets each attribute
back - a slot through its descriptor, which a frozen dataclass's `__setattr__` does not
see, and anything else into `__dict__`. Nothing else of the class runs: not `__init__`,
not `__post_init__`, not `__getstate__` / `__setstate__`. Methods are never stored; they
come from the code as it is when the record is loaded.

A value that cannot be rebuilt that way - a function, a class, a C type with state of
its own such as an open file or a lock - is refused with `SerializationError`, and so is
an object holding one as an attribute. See
[Limitations](limitations.md#objects-are-recorded-by-their-attributes) for what this
couples a record to.

## Portability

Every path inside a record's JSON is relative to that record's own `records/` directory
(`"../data/arr-3f9a1c2b7e6d4a10.npy"`). Absolute paths are never written. Consequently:

- a whole store directory can be moved, copied, or checked out anywhere;
- a single identity directory can be moved on its own and still resolves;
- records written on one machine load on another.

What is *not* portable is a record whose values reference classes by name: a recorded
custom object stores `{"module": ..., "qualname": ...}` and is re-imported on load. If
that class has since moved or been deleted, loading raises `IdentityResolutionError`.
This is also why classes defined in `__main__`, or inside a function, are refused at
record time rather than failing mysteriously later.

## Run manifests

A record says what a call did, not what it was made with, and since its id is derived
from its behavior, re-recording identical behavior writes nothing. A
`session(store, context=..., repos=..., packages=...)` block keeps that history beside
the records without touching them:

```
tests/regression_data/
  _runs/
    run-20260926T104527.962591Z-6d6be6.json
  myapp/...
```

Each manifest lists every record the session saw, as `new` (written by it) or
`matched` (the same behavior was already on disk), together with:

- the interpreter, platform and this library's version;
- for each named distribution, its version and install source - the commit, for a git
  install;
- for each named repository, `HEAD` and whether tracked files are modified;
- the caller's `context`: anything else identifying the run, such as a data version or
  a seed.

Run ids start with a UTC timestamp to the microsecond, so manifests sort by start time.
Replay and `prune` only look inside `records/` and `data/` directories and never read
`_runs/`; manifests can be committed or deleted without affecting any record.
`ReplayReport.save(path, environment=...)` writes the other half of the history: a
replay's results beside the environment it ran in.

## Trust

A store is data, but loading one is not inert, so treat it as you treat the code it
sits beside and review changes to it:

- a recorded object names its class, and loading it **imports that module** - running
  its top-level code - then allocates an instance with `object.__new__` and sets the
  recorded attributes on it. No method of the class is called, but which class is the
  record's choice;
- a side file is not checked against its name on load by default, so a modified or
  swapped `.npy` / `.parquet` loads as if it were the recorded value.
  `RecordStore(root, verify_side_files=True)` checks every side file it loads and raises
  `CorruptRecordError` on a mismatch (checking a pickle before it is unpickled), at the
  cost of one extra hash per file;
- every side-file reference must resolve inside the identity directory. An absolute
  path, or one that climbs out with `..`, raises `CorruptRecordError` rather than
  pointing a file reader somewhere else;
- nothing is pickled or unpickled unless you opt in. `np.load` runs with
  `allow_pickle=False`, and the pandas serializers refuse both to write and to read a
  pickle unless built with `allow_pickle=True` - a record's `format` field cannot ask
  for it by itself.

## Deferred writes

Computing an invocation's `input_hash` requires the serialized argument tree - which
means arguments are serialized *before* it is known whether anything will be kept. A
call already recorded in this process, or one that raised with
`record_exceptions=False`, writes no record at all.

So `SerializationContext(deferred=True)` buffers side-file writes instead of performing
them. The engine calls `flush()` only once a record is actually committed, and
`discard()` otherwise, so a skipped recording leaves no unreferenced `.npy` or
`.parquet` files in `data/`.

## The record format

```json
{
  "schema": { "schema_version": 3, "library_version": "0.4.0" },
  "record_id": "0f3a9c2e...",
  "input_hash": "3f9a1c2b7e6d4a10...",
  "created_at": "2026-08-22T09:14:03.221847+00:00",
  "identity": {
    "module": "myapp.pricing",
    "qualname": "compute_score",
    "kind": "function",
    "code_hash": "d4cbbd9618a00aa7",
    "signature": "(frame, weights=1.0)"
  },
  "invocation": {
    "args": [{ "__type__": "polars.DataFrame", "path": "../data/df-9c1a....parquet", "shape": [1000, 4] }],
    "kwargs": { "weights": 1.0 }
  },
  "output": { "__type__": "numpy.ndarray", "path": "../data/arr-3f9a....npy", "shape": [1000], "dtype": "float64" },
  "tags": {}
}
```

Written with `indent=2, sort_keys=True`, so a new record reads well in review and a
`git diff` is never a reordered blob.

`"__type__"` is the reserved tag naming the serializer that wrote a value and will read
it back. A dict being recorded that contains that key literally is rejected with a clear
error rather than silently corrupting the tree.

A record is strict JSON (RFC 8259), so `jq` and parsers outside Python read it. JSON has
no literal for NaN or infinity, so a non-finite float is tagged:
`{"__type__": "float", "value": "nan"}` (or `"inf"`, `"-inf"`). A numpy scalar keeps its
usual tag and stores the same strings as its `value`. Finite floats stay plain numbers.

Optional fields are **absent** rather than `null` when unset, which is what lets a
recorded return value of `None` (`"output": null`) be told apart from a call that raised
and has no output at all (no `"output"` key).

- `output` and `exception` are mutually exclusive.
- `state_before` / `state_after` appear only for methods. They hold the recorded
  attributes only; which attributes those were (`state_attrs`, `exclude_attrs`) and how
  to rebuild the receiver (`factory`) are not stored, and replay takes them from the
  decorator as it is now.
- `exception` holds `exc_type`, `message`, and `traceback_text`. `exc_type` is the
  module-qualified class name (`"builtins.ValueError"`, `"requests.exceptions.ConnectionError"`),
  so same-named exceptions from different modules stay distinct. Only `exc_type` is
  compared - messages routinely carry nondeterministic detail like addresses, paths, and
  timings - and the traceback is retained purely for display.

### Schema versions

| Version | Change |
|---|---|
| 1 | Initial format. `exc_type` is the bare class name; non-finite floats are bare `NaN` / `Infinity` tokens. |
| 2 | `exc_type` is module-qualified; non-finite floats are tagged, so the file is strict JSON. |
| 3 | Objects are stored by their attributes (`"attrs"`), named tuples included (their fields by name); `mappingproxy` and dicts with non-string keys are tagged values. |

Version 2 records load and compare with no migration: an object's `"state"` (its
`__getstate__()` then) is set back as its attributes, and a named tuple tagged
`"namedtuple"` is rebuilt from its positional `"items"`. Re-recording a call whose
arguments hold an object or a named tuple writes a new record beside the old one,
because the serialized form - and so the input hash - changed. Version 1 records load
and compare with no migration: bare `NaN` / `Infinity` tokens
are still read, and a bare `exc_type` is compared against the bare class name of what
was raised (so it cannot tell same-named exceptions apart). A record from a newer
version than the installed library reads is refused with `CorruptRecordError`.

## Loading

Reading a record does not deserialize it. `RecordStore.load_record` gives you the
metadata; the values are materialized on demand:

```python
store = RecordStore("tests/regression_data")
path = store.iter_record_paths()[0]
record = store.load_record(path)
ctx = store.deserialization_context(record.identity)

record.input_hash  # free
record.identity.qualname  # free
record.load_output(ctx)  # now the .npy is read
```

This is what lets replay select records (`select=`) and name test cases without touching
the parquet files a record points at.

## Limitations

- **No locking.** Several processes may record to one store at once: identical
  recordings converge on one file and changed ones add records, and every write is
  atomic. Pruning is the exception - see below.
- **No schema migration tooling.** Older records are read as they are (see
  [Schema versions](#schema-versions)); nothing rewrites them to the current version.
- **No cross-identity deduplication.** The same array recorded for two different
  functions is stored twice, on purpose - see the portability guarantee above.
- **Pruning is manual.** A `data/` file stops being referenced when the record pointing
  at it is removed, or a save is interrupted, and nothing collects it
  automatically. `store.prune()` lists such files - every identity, or one with
  `store.prune(identity)` - and `store.prune(dry_run=False)` deletes them. A file is kept
  while its name appears in any record file of its identity, so a record that no longer
  loads still protects its data. Records themselves are never pruned, and a store should
  not be pruned while something is recording to it.
