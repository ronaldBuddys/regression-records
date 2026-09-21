# Limitations

What the package does not do, and what to watch for. The
[gotchas example](auto_examples/13_gotchas.rst) shows most of these running. Open work is
tracked in the [issue tracker](https://github.com/ronaldBuddys/regression-records/issues).

## Ways a replay can pass without testing anything

A regression tool failing open is the failure that matters most, so these come first.

- **Replay checks records, not functions.** A decorated function that was never called
  while recording was on, or whose records were deleted, has nothing to replay and does
  not appear in the report. Nothing warns that it went unchecked.
- **Recording is off unless switched on.** A run meant to record, started without
  `REGRESSION_RECORDS=1`, writes nothing and leaves the store as it was.
- **An empty store passes.** A store path that does not exist raises
  `FileNotFoundError`, but a directory that exists and holds no records - freshly
  created, wiped, or git-ignored so CI checks out an empty one - replays nothing:
  `replay()` passes with zero cases, and `replay_tests` gives pytest's "empty parameter
  set" skip. Check `len(report.cases)` if that matters.
- **A record made from the wrong behavior passes against it.** Replay compares with
  what was recorded, not with what is correct. Review a store's diff when records are
  added.

## Recording

- **Records are never replaced or deleted automatically.** Recording changed behavior
  adds a record beside the old one, which then fails replay until you delete its file
  by hand; `RecordStore.prune(dry_run=False)` then removes the data files it leaves
  behind.
- **"Identical" means identical once serialized.** A recording is skipped only when its
  arguments, its output (or exception type) and, for a method, its state serialize
  exactly as an earlier record's. A result that moved in the last bit - within any
  tolerance replay would apply - is a new record.
- **Within one process a call is recorded once**, keyed on its arguments (and, for a
  method, the receiver's state before the call), not on its result. A function that
  returns something different the second time it is called with the same arguments in
  the same process has only its first result recorded. Redefining the decorated
  function (in a notebook, say) records again, but a change in a helper it calls does
  not: re-record an intended change of that kind from a fresh process.
- **Several processes may record to one store at once**, as under `pytest -n auto`:
  identical recordings converge on one filename, and changed ones add records. Do not
  run `RecordStore.prune` while anything is recording, because a data file is written
  before the record that references it.
- **Re-recording after a library upgrade may add records.** A DataFrame or Series
  *argument* is hashed from its parquet bytes, which change across some library
  releases (polars changed them between 1.0 and 1.20, and again between 1.30 and 1.40).
  The same call then gets a new input hash, so re-recording it adds a record rather
  than converging with the old one. Both replay, since the input hash only groups
  records of a call and nothing is looked up by it; delete whichever you no longer
  want. Adding a parameter with a default to a function does the same.
- **Record files are ordinary files**, created with the process's default permissions,
  so they are as readable as your umask makes them. Every recorded argument is stored
  in full: do not record calls whose arguments carry secrets.
- **`record_method` beneath `@staticmethod` cannot be detected:** the first parameter is
  taken for the receiver and dropped from the recorded arguments. A staticmethod has no
  state to snapshot; use `record_func`.
- **An unrecognised `REGRESSION_RECORDS` value raises `ValueError`** on the next
  decorated call, rather than leaving recording silently off.

## Replay

See [Replay](replay.md#what-replay-cannot-see) for more.

- **A call is replayed from its recorded inputs alone.** A function that reads a clock,
  a database, an environment variable or a global sees today's value, not the recorded
  one.
- **Side effects happen again**: a replayed function that writes files or sends
  requests does so again.
- **The function must be importable by name.** Lambdas, closures and functions defined
  in `__main__` are recorded today but replay as `not_importable`; a later release will
  refuse them at decoration time. A function defined in `__main__` is also stored
  under `__main__/`, so the same function run as a script and under pytest has two
  identities, and `<lambda>` and `<locals>` appear in directory names, which Windows
  does not allow.
- **Method records are not replayed yet** (`not_replayable`). The next release rebuilds
  `self` from the recorded state, or through a `factory=` when only part of the state is
  recorded.

## Values that cannot be recorded

Each raises `SerializationError` when recording, rather than writing a record that
cannot be read back.

- `dtype=object` numpy arrays, structured arrays with an object field, and `ndarray`
  subclasses such as `MaskedArray` and `matrix` (only the data buffer would be saved).
- A polars `LazyFrame`: collection can be expensive or side-effecting, so `.collect()`
  is yours to call.
- A polars column of dtype `Object`, the one polars dtype parquet cannot hold. Cast it
  (a `Struct` for records, `String` for labels) before returning the frame. There is no
  pickle opt-in for polars as there is for pandas.
- A pandas frame Arrow cannot represent (a mixed-type object column), unless you opt in
  to pickling that one file with `PandasDataFrameSerializer(allow_pickle=True)`.
- Dicts with non-string keys, and dicts using the reserved key `__type__`.
- Objects with no dict-shaped state (`__slots__`-only classes), classes defined in
  `__main__` or inside a function, and anything that refers back to itself - a record is
  a tree. Leave a back-reference out with `exclude_attrs`.
- Generator and async functions are refused when decorated: calling one returns a
  generator or coroutine, not a result.
- A combination of `Flag` members (`Perm.R | Perm.W`), which has no single member name;
  record its `.value`. A subclass of a `pathlib` class goes to the generic object
  serializer like any other class. Numpy complex scalars are refused (complex arrays and
  Python `complex` are fine).

## Values that record, but lose something

- A concrete `pathlib.Path` loads as the loading platform's `Path`, so a `WindowsPath`
  recorded on Windows becomes a `PosixPath` on Linux. The pure path classes keep theirs.
- A `pandas.Timestamp` keeps its zone by name for `ZoneInfo` and pytz zones; any other
  comes back as its fixed UTC offset.
- An `IntEnum` / `StrEnum` member is recorded as its `int` / `str` value, and a `dict`,
  `list` or `tuple` subclass as the plain builtin. (A named tuple keeps its class.)
- A `datetime` keeps a `ZoneInfo` zone and its `fold`, but any other `tzinfo` (pytz,
  dateutil) comes back as its fixed UTC offset. Datetimes compare as instants, so a
  change of time zone alone is not reported.
- `DatetimeIndex.freq` does not survive parquet.
- On polars older than 1.35, a `Decimal` column whose precision was inferred rather
  than declared has precision `None`, and reads back from parquet as 38, so the record
  fails replay as a `schema_mismatch`. Declare it (`pl.Decimal(18, 4)`) or upgrade.
- A `np.longdouble` is only as precise as the platform that loads it: an 80-bit value
  recorded on x86 Linux loses digits where `longdouble` is `float64` (Windows, Apple
  silicon).

## Comparison

- **Integers are always compared exactly**, so there is no way to give an integer
  result a tolerance; return it as a float if it is a measurement.
- **Exceptions in schema version 1 records are compared by bare class name**, so
  same-named exceptions from different packages are indistinguishable there. Records
  written since store the module-qualified name.
- Sets are compared by membership, so tolerance cannot apply inside one.

## The store

- **Writes are atomic**, so a reader never sees a half-written file; see
  [Recording](#recording) for several writers at once.
- **Loading is not safe on a store you do not trust** - see
  [Trust](persistence.md#trust).
- `-0.0` and `0.0` are different arguments.
- Side files are now named by a 32-character hash (16 before). An array or frame
  argument's serialized form holds that name, so its input hash changed too:
  re-recording such a call adds a record beside the earlier one. The earlier record and
  its data file stay loadable and still replay.
- Schema version 2 tags non-finite floats (see
  [the record format](persistence.md#the-record-format)), which changes the input hash
  of a call whose arguments hold a NaN or infinity, with the same effect. The version 1
  record stays loadable and still replays.
- A schema version 1 record stores an exception's bare class name, so it cannot tell
  same-named exceptions from different modules apart. Re-record to get qualified names.
- Data files no longer referenced by any record accumulate when records are removed,
  until `RecordStore.prune` is run by hand.
- Moving a function between modules is not auto-resolved - it is ambiguous in general.
  Its records replay as `not_importable` until `RecordStore.rename_identity(old, new)`
  moves them by hand.
- There is no CLI for browsing or pruning a store; it is JSON in directories, and
  ordinary tools work on it (and `RecordStore.prune` from Python).
