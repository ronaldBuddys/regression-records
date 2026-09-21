# Changelog

## 0.2.0

Recording and checking are now two separate steps. The decorators only record, and
only when switched on. `replay()` checks the records later, against whatever the code
and environment are by then.

### Added

- `replay(store, ...)` reruns every recorded call against the current code and returns
  a `ReplayReport`. Each record gets its own `ReplayCase`, with a status of `passed`,
  `differs`, `raised`, `not_importable`, `not_replayable` or `not_loadable`, the diffs,
  and a code-drift flag. Tolerances are passed to `replay()`: `tolerance` for every
  case, and `tolerance_overrides` for particular functions, keyed by a glob on
  `"module:qualname"`.
- **Method replay.** `replay()` replays `record_method` records: it imports the
  method's class, rebuilds the receiver in its recorded state before the call, calls
  the method, and compares the return value (or exception) and the state after
  (`$.state_after...`). From a whole recorded state the receiver is rebuilt as
  unpickling does, with `cls.__new__` and then `__setstate__` or `__dict__.update`,
  so `__init__` is not called.
- `record_method(..., factory=...)`: a zero-argument callable that builds the receiver
  for replay, which then sets the recorded attributes on it. A factory returning an
  instance of another class, or a rebuild that raises, gives `not_replayable`.
- `replay_tests(store)` turns a store into a pytest suite with one test per record:
  `test_regressions = replay_tests("tests/regression_data")`.
- The recording switch. Set the environment variable `REGRESSION_RECORDS=1`, call
  `enable()` / `disable()`, or use a `with recording():` block. Recording is off by
  default, so a decorated function costs one check per call.
- `RecordStore.iter_record_paths()`, `RecordStore.load_record(path)`,
  `FunctionIdentity.resolve()` and `RecordStore.prune()`.

### Changed

- **Records are never replaced.** A record's id is now derived from what the call did.
  Recording the same behavior again writes nothing, including from parallel pytest-xdist
  workers, which all land on the same file. Recording changed behavior adds a new record
  beside the old one, and the old record then fails replay until you delete it by hand.
- **A tolerance override inherits every field it does not set.** A `path_overrides`
  entry that gives only `rtol` now keeps its parent's `atol`, `equal_nan` and
  `strict_dtype`, where it used to reset them to the defaults. Every matching override
  applies, longest last.
- **`Serializer.load` receives only the payload** that `store()` returned. The
  `"__type__"` key is no longer included. Custom serializers that read it must stop
  doing so.
- **Every record holds exactly one of `output` and `exception`.** A record file with
  both or neither is reported as corrupt.
- **Integers compare exactly, whatever the tolerance.** Tolerance still applies once a
  float is involved.
- **`state_attrs` and `exclude_attrs` need `factory=`.** A record of part of an
  object's state cannot rebuild the object, so `record_method` raises `TypeError` when
  either is given without a factory.
- **Callables replay could not import are refused when decorated**, with a
  `TypeError`: lambdas, functions and classes defined inside a function, a
  `functools.partial` or other callable without a `__qualname__`, and anything defined
  in `__main__`. They used to be recorded and then replay as `not_importable`, which
  records made by earlier versions still do. Move such functions into an importable
  module.

### Removed

- `RunMode`, `OnNewCase`, `OverwritePolicy` and `OnMismatch`, together with the
  decorators' `mode=`, `tolerance=`, `on_new_case=`, `overwrite=`, `on_mismatch=` and
  `comparator_registry=` arguments. Nothing compares while a decorated call runs.
- The decorators' `name=` argument. Replay imports a function by its real module and
  qualname, so a record filed under any other name could never be replayed.
- `wrapper.last_result` and `wrapper.config`. Results now live in the `ReplayReport`.
- `regression_records.config` and `regression_records.matching`. `compute_input_hash`
  and `canonical_payload` moved to `regression_records.records`.
- `RecordStore.find_by_input_hash`. Nothing is looked up by input hash any more; it
  only groups the records of one call.
- `RecordNotFoundError`, `UnmatchedInvocationError` and `RegressionMismatchError`.

### Migrating from 0.1

- **Checking a run against its records:** replace validate-in-place with
  `replay(store)` or a `replay_tests` module.
- **Accepting an intended change:** re-record with `REGRESSION_RECORDS=1`, then delete
  the old record file and run `RecordStore.prune(dry_run=False)`. This replaces
  `overwrite=IF_MISMATCH`.
- **Recording in CI:** set `REGRESSION_RECORDS=1` for the recording job. This replaces
  `mode=RECORD_ONLY`.
- **`record_method` with `state_attrs` or `exclude_attrs`:** add `factory=`, a
  zero-argument callable returning an instance.
- **Functions decorated in a script or notebook, or inside another function:** move
  them to a module and import them.
- **Existing stores work unchanged.** Schema version 1 and 2 records replay with no
  migration.
