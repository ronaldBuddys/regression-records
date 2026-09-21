# Limitations

What the package does not do, and what to watch for. The
[gotchas example](auto_examples/13_gotchas.rst) shows most of these running. Open work is
tracked in the [issue tracker](https://github.com/ronaldBuddys/regression-records/issues).

## Ways a suite can pass without testing anything

A regression tool failing open is the failure that matters most, so these come first.

- **The default mode records whatever it has not seen.** Under `RunMode.BOTH` a call
  with no history is recorded, and passes. If the store directory is missing,
  git-ignored or freshly wiped, *every* call is new and the suite is green. Run CI
  under `RunMode.VALIDATE_ONLY`, where a call with no record is an error.
- **A signature change orphans history silently.** Arguments are bound against the
  signature before hashing, so adding a parameter - even with a default - gives every
  call a new input hash: a new case, recorded, green. The old records are not reported
  as unreachable.
- **`RunMode.RECORD_ONLY` with the default `OverwritePolicy.NEVER` does nothing for a
  case that is already recorded:** it is neither re-recorded nor validated, and
  `last_result` stays `None`.
- **`overwrite=IF_MISMATCH` left in place accepts everything.** It is meant for one run.
  There is no command-line or environment switch for it yet, so re-baselining means
  editing the decorator and editing it back.
- **`record_method` keys on arguments, not on the receiver.** Two instances in different
  states calling a method with the same arguments are the *same case*; the second is
  reported as a state mismatch, not recorded as its own case.
- **`record_method` beneath `@staticmethod` cannot be detected:** the first parameter is
  taken for the receiver and dropped from the input hash. A staticmethod has no state to
  snapshot; use `record_func`.

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
  fails validation as a `schema_mismatch`. Declare it (`pl.Decimal(18, 4)`) or upgrade.
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

- **Not safe for concurrent writers.** Writes are atomic, so a reader never sees a
  half-written file, but two processes recording the same new case (as under
  `pytest -n auto`) each write a record, and which one later runs match is not
  deterministic. Record from one process; validating in parallel is fine.
- **Loading is not safe on a store you do not trust** - see
  [Trust](persistence.md#trust).
- `-0.0` and `0.0` are different arguments.
- Side files are now named by a 32-character hash (16 before). An array or frame
  argument's serialized form holds that name, so its input hash changed too: a case
  recorded before the change with such an argument is recorded again as new rather than
  validated. The earlier record and its data file stay loadable.
- Schema version 2 tags non-finite floats (see
  [the record format](persistence.md#the-record-format)), which changes the input hash
  of a call whose arguments hold a NaN or infinity. Such a case recorded under version 1
  is recorded again as new rather than validated; the version 1 record stays loadable.
- A schema version 1 record stores an exception's bare class name, so it cannot tell
  same-named exceptions from different modules apart. Re-record to get qualified names.
- A DataFrame or Series *argument* is matched by the hash of its parquet bytes, which
  change across some library releases (polars changed them between 1.0 and 1.20, and
  again between 1.30 and 1.40). After such an upgrade, calls taking a frame are recorded
  as new cases rather than validated; re-record, and see the gotchas example.
- Data files no longer referenced by any record accumulate when records are
  overwritten or removed, until `RecordStore.prune` is run by hand.
- Moving a function between modules is not auto-resolved - it is ambiguous in general.
  `RecordStore.rename_identity(old, new)` moves the history by hand.
- A function defined in `__main__` is stored under `__main__/`, so the same function run
  as a script and under pytest has two identities. `<lambda>` and `<locals>` appear in
  directory names, which Windows does not allow.
- There is no CLI for browsing or pruning a store; it is JSON in directories, and
  ordinary tools work on it (and `RecordStore.prune` from Python).
