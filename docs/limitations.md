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
- Not yet supported, though the comparison layer would handle them: `frozenset`,
  `bytearray`, `datetime.time`, `datetime.timedelta`, `Decimal`, `pathlib.Path`,
  `complex`, and plain `Enum` members.

## Values that record, but lose something

- An `IntEnum` / `StrEnum` member is recorded as its `int` / `str` value, and a `dict`,
  `list` or `tuple` subclass as the plain builtin. (A named tuple keeps its class.)
- `datetime` keeps its UTC offset but not its `ZoneInfo` or `fold`.
- `DatetimeIndex.freq` does not survive parquet.
- A `np.longdouble` scalar becomes a Python float.
- A pandas categorical column is compared by its labels; a change to the category set
  alone is not reported.

## Comparison

- **Integers are compared within tolerance**, like floats: with the default `rtol` a
  count of 1,000,000,000 matches 1,000,000,050. Give counts and identifiers an exact
  override. Integers beyond 2**53 are always compared exactly.
- **polars `null` and `NaN` compare equal** in a float column under the default
  `equal_nan=True`, because both become NaN on the way to numpy.
- **Exceptions are compared by bare class name**, so same-named exceptions from
  different packages are indistinguishable.
- Sets are compared by membership, so tolerance cannot apply inside one.

## The store

- **Not safe for concurrent writers.** Writes are atomic, so a reader never sees a
  half-written file, but two processes recording the same new case (as under
  `pytest -n auto`) each write a record, and which one later runs match is not
  deterministic. Record from one process; validating in parallel is fine.
- **Loading is not safe on a store you do not trust** - see
  [Trust](persistence.md#trust).
- Record JSON may contain bare `NaN` / `Infinity` tokens, which Python reads and
  strict JSON parsers reject. `-0.0` and `0.0` are different arguments.
- A DataFrame *argument* is matched by the hash of its parquet bytes, which may change
  across pyarrow or polars versions; such calls would then be recorded as new cases.
- Data files no longer referenced by any record accumulate when records are
  overwritten; nothing prunes them.
- Moving a function between modules is not auto-resolved - it is ambiguous in general.
  `RecordStore.rename_identity(old, new)` moves the history by hand.
- A function defined in `__main__` is stored under `__main__/`, so the same function run
  as a script and under pytest has two identities. `<lambda>` and `<locals>` appear in
  directory names, which Windows does not allow.
- There is no CLI for browsing or pruning a store; it is JSON in directories, and
  ordinary tools work on it.
