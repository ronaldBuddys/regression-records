# Limitations

What the package does not do, and what to watch for. The
[gotchas example](auto_examples/05_gotchas/15_gotchas.rst) shows most of these running. Open work is
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
- **A setting the function reads, rather than receives, is not in its record.** A
  module-level global, or an environment variable not named in `env=`, has whatever
  value the replaying process gives it. Records made and replayed under one setting
  say nothing about another, and a change that shows only under the recorded setting
  passes on a machine set differently. Pass the setting as an argument, or name the
  variable in `env=` ([Replay](replay.md#the-environment-a-call-reads)).

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
  function (reloading its module, say) records again, but a change in a helper it calls does
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
  so they are as readable as your umask makes them. Every recorded argument, return
  value, method state and exception message is stored in full, as are the variables
  named in `env=`: do not record calls that carry secrets in any of them. Build a
  client from its key at the edge, and record the functions beneath it; a method's
  client can be left out with `exclude_attrs`. A key the function reads from the
  environment can be named in `secret_env=`, which stores a fingerprint, not the
  value; a short or guessable secret can be found from its fingerprint by trying
  candidates.
- **`record_method` beneath `@staticmethod` cannot be detected:** the first parameter is
  taken for the receiver and dropped from the recorded arguments. A staticmethod has no
  state to snapshot; use `record_func`.
- **An unrecognised `REGRESSION_RECORDS` value raises `ValueError`** on the next
  decorated call, rather than leaving recording silently off.

## Replay

See [Replay](replay.md#what-replay-cannot-see) for more.

- **A call is replayed from its recorded inputs alone.** A function that reads a clock,
  a database, a global or an environment variable not named in `env=` sees today's
  value, not the recorded one.
- **`env=` covers what is read during the call.** A variable read at import, into a
  module constant, was read before replay set it. Replay sets variables in the
  process-wide `os.environ`, so replaying from several threads at once is unsafe.
- **A secret named in `secret_env=` cannot be set on replay**, only compared: the call
  runs under today's value, and `case.secret_env_changed` says when that differs.
- **Side effects happen again**: a replayed function that writes files or sends
  requests does so again.
- **The function must be importable by name.** A lambda, a function or class defined
  inside another function, a `functools.partial`, and anything defined in `__main__` (a
  script being run, or a notebook) are refused with a `TypeError` when decorated. That
  includes a method of a class defined there. Records of such functions written by
  earlier versions replay as `not_importable`.
- **A method's object is rebuilt without `__init__`.** From a whole recorded state,
  replay calls `object.__new__(cls)` and sets the recorded attributes back, as for any
  recorded object, so an invariant `__init__` establishes without storing it as an
  attribute - registering the object somewhere, opening a resource - does not hold on
  replay. With `state_attrs` or `exclude_attrs` the record holds only part of the state,
  so `factory=` is required and builds the object instead.
- **How a receiver is rebuilt is not recorded.** Replay uses the current decorator's
  `state_attrs`, `exclude_attrs` and `factory`. Changing `state_attrs` or
  `exclude_attrs` after recording makes old records differ on the attributes that moved
  in or out of the captured state; re-record them.

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
- Dicts using the reserved key `__type__`. (Dicts with non-string keys record, as a
  list of key/value pairs that keeps the keys' types.)
- Values that are not data and have no attributes to rebuild them from: functions,
  classes, modules, and C types holding state of their own - open files, locks,
  exceptions, `types.SimpleNamespace`, objects from compiled extensions (a jax array, a
  QuantLib date). An object holding one as an attribute is refused as a whole; the
  attribute is never skipped. Record the data instead, or register a leaf `Serializer`.
- Classes defined in `__main__` or inside a function, and anything that refers back to
  itself - a record is a tree. Leave a back-reference out with `exclude_attrs`.
- Subclasses of a C type that `object.__new__` cannot allocate - a `tuple` subclass
  other than a named tuple, say - are recorded as the builtin (see below), and one
  defining `__getnewargs__` / `__reduce_ex__` to rebuild itself gets no special
  treatment.
- Generator and async functions are refused when decorated: calling one returns a
  generator or coroutine, not a result.
- A combination of `Flag` members (`Perm.R | Perm.W`), which has no single member name;
  record its `.value`. A subclass of a `pathlib` class is refused: stored by its
  attributes it would record CPython internals. Numpy complex scalars are refused
  (complex arrays and Python `complex` are fine).

## Objects are recorded by their attributes

- **`__getstate__` / `__setstate__` are ignored**, and so are `__init__` and a
  dataclass's `__post_init__` when a record is loaded: an object is its `__dict__` plus
  every slot set in its class hierarchy, rebuilt with `object.__new__`. A validation or
  normalization only those methods perform is not re-run on a loaded object.
- **A record names attributes, private ones included.** Rename an attribute and records
  written before the rename no longer fit the class. A removed slot, or an added
  dataclass or named-tuple field, fails loading with a message saying so (replay reports
  `not_loadable`); for a plain `__dict__` class the old name is set back as it was, and
  the difference shows up as a named diff, or as the replayed call raising. Re-record.
- **Caches are attributes too.** A lazily filled cache is recorded and compared like any
  other attribute, so a change in which lookups a function performs can show up as a
  diff inside the cache.

## Values that record, but lose something

- A concrete `pathlib.Path` loads as the loading platform's `Path`, so a `WindowsPath`
  recorded on Windows becomes a `PosixPath` on Linux. The pure path classes keep theirs.
- A `pandas.Timestamp` keeps its zone by name for `ZoneInfo` and pytz zones; any other
  comes back as its fixed UTC offset.
- An `IntEnum` / `StrEnum` member is recorded as its `int` / `str` value, and a `dict`,
  `list` or `tuple` subclass as the plain builtin. (A named tuple keeps its class.)
- Read-only numpy arrays, and broadcast views, load as ordinary writeable arrays.
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
