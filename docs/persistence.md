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
inside JSON.

## Identity: which directory a call belongs to

A record's identity is `(module, qualname)`, taken from the *undecorated* function at
decoration time.

`__qualname__` already encodes class nesting - `Portfolio.rebalance`, `Outer.Inner.run` -
which is what keeps two same-named methods on different classes from colliding, a
distinction a bare method name loses.

Each record also stores a **`code_hash`**: a hash of the function's source, normalized to
ignore whitespace and comments (falling back to the compiled bytecode when source is
unavailable, and to the literal string `"unknown"` when even that is). It is
**informational only** - never part of the directory path, never part of the matching
key. That is deliberate: a refactor must not orphan a function's history. What it buys
you is `CodeDriftWarning`, which tells you whether a passing comparison means "behavior
held across a real change" or merely "nothing changed".

`signature` is stored for the same reason - human debugging - and is never compared.

### Moving a function

Not auto-resolved, because it is genuinely ambiguous: a function that vanished from one
module and a similar one that appeared in another are not necessarily the same function.
`RecordStore.rename_identity(old, new)` does it by hand:

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

## Which record a call matches

Matching is on **`input_hash`**: a sha256 over the canonicalized *serialized* arguments -
`json.dumps(..., sort_keys=True, separators=(",", ":"))` of `{"args": ..., "kwargs":
...}`.

Serializing first is what makes the key meaningful. By that point every external value is
already content-addressed, so two equal numpy arrays contribute identically to the hash
regardless of object identity, in this process or any other.

Arguments are normalized through `inspect.signature().bind()` plus `apply_defaults()`
before hashing, so `f(2, 3)`, `f(2, b=3)`, `f(a=2, b=3)` and `f(2)` with `b` defaulting to
`3` are all **one** recorded case rather than four.

For methods, `self` is excluded from the recorded arguments and from the hash; the
receiver is described by the captured state instead. `args` is therefore always `[]` for
a method: once the receiver is gone, positional order no longer describes the call, and
`apply_defaults()` has already named everything.

**Matching is always exact.** Tolerances apply to a call's output and state, never to its
inputs: two invocations with slightly different arguments are two cases to record, not
one case that nearly matches.

The function's identity is deliberately *not* mixed into the hash - records are already
namespaced by identity on disk, and including it would change every hash whenever a
function moved module.

## File naming

**Records** are `rec-<input_hash[:16]>-<record_id[:8]>.json`. The input hash leads so
that finding the record for a call is a directory glob rather than a scan of every
historical record - it stays cheap as a function accumulates cases, and it runs on every
validated call. The truncated hash in the name is confirmed against the full one inside
the file before a record is accepted as a match. `record_id` is a uuid4 hex, and follows
so that distinct recordings of the same inputs cannot collide.

**Data files** are `<prefix>-<sha256[:16]>.<ext>`, hashed over the file's own content:

| Prefix | Type | Format |
|---|---|---|
| `arr` | `numpy.ndarray` | `.npy`, via `np.save(..., allow_pickle=False)` |
| `df` | `polars.DataFrame` | `.parquet` |
| `series` | `polars.Series` | `.parquet`, stored as a one-column frame |
| `pddf` | `pandas.DataFrame` | `.parquet`, or `.pkl` on fallback |
| `pdseries` | `pandas.Series` | as above, stored as a one-column frame |

Because the name *is* the content hash, a file that already exists is known to hold
identical content and is not rewritten. Deduplication is scoped per identity directory,
not globally: one function's history stays a self-contained unit you can move, copy, or
delete without consulting the rest of the store.

Numpy scalars (`np.float64`, `np.datetime64`, ...) are stored **inline** in the JSON with
their dtype, not as side files - they are small, and being able to read them in the
record is worth more than uniformity.

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

## Trust

A store is data, but loading one is not inert, so treat it as you treat the code it
sits beside and review changes to it:

- a recorded custom object names its class, and loading it **imports that module** -
  running its top-level code - then hands the recorded state to the class's
  `__setstate__`. Only a class is ever instantiated, but which class is the record's
  choice;
- every side-file reference must resolve inside the identity directory. An absolute
  path, or one that climbs out with `..`, raises `CorruptRecordError` rather than
  pointing a file reader somewhere else;
- nothing is pickled or unpickled unless you opt in. `np.load` runs with
  `allow_pickle=False`, and the pandas serializers refuse both to write and to read a
  pickle unless built with `allow_pickle=True` - a record's `format` field cannot ask
  for it by itself.

## Deferred writes

Computing an invocation's `input_hash` requires the serialized argument tree - which
means arguments are serialized *before* it is known whether anything will be kept. In
validate-only mode, or on a mismatch under `OverwritePolicy.NEVER`, no record is written
at all.

So `SerializationContext(deferred=True)` buffers side-file writes instead of performing
them. The engine calls `flush()` only once a record is actually committed, and
`discard()` otherwise. Without this, a CI run that validates and writes nothing would
still litter `data/` with unreferenced `.npy` and `.parquet` files.

## The record format

```json
{
  "schema": { "schema_version": 1, "library_version": "0.1.0" },
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

Written with `indent=2, sort_keys=True`, so a re-record produces a readable `git diff`
rather than a reordered blob.

`"__type__"` is the reserved tag naming the serializer that wrote a value and will read
it back. A dict being recorded that contains that key literally is rejected with a clear
error rather than silently corrupting the tree.

Optional fields are **absent** rather than `null` when unset, which is what lets a
recorded return value of `None` (`"output": null`) be told apart from a call that raised
and has no output at all (no `"output"` key).

- `output` and `exception` are mutually exclusive.
- `state_before` / `state_after` appear only for methods.
- `exception` holds `exc_type`, `message`, and `traceback_text`. Only `exc_type` is
  compared - messages routinely carry nondeterministic detail like addresses, paths, and
  timings - and the traceback is retained purely for display.

## Loading

Reading a record does not deserialize it. `RegressionRecord.from_dict` gives you the
metadata; the values are materialized on demand:

```python
record = store.find_by_input_hash(identity, input_hash)
ctx = store.deserialization_context(identity)

record.input_hash  # free
record.identity.qualname  # free
record.load_output(ctx)  # now the .npy is read
```

This is what keeps `find_by_input_hash` cheap: it compares hashes without ever touching
the parquet files a record points at.

## Limitations

- **No concurrency control.** Two processes recording to one store can interleave writes.
  One process at a time.
- **No schema migration tooling.** `schema_version` is stamped on every record so a
  future version can recognize an old one; nothing reads it yet.
- **No cross-identity deduplication.** The same array recorded for two different
  functions is stored twice, on purpose - see the portability guarantee above.
- **No pruning tool.** A `data/` file stops being referenced when the record pointing at
  it is overwritten; nothing collects it. The store is plain files, so `git` and ordinary
  shell tools are the intended answer.
