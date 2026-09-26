# Changelog

## 0.4.0

An object of any class is now recorded, compared and rebuilt as its attributes, so a
class of your own never needs a serializer. Serializers are for leaf values only.

### Changed

- **Objects are stored by their attributes**: their `__dict__` plus every slot set
  anywhere in their class hierarchy, recursively. `__slots__` classes, frozen and
  slotted dataclasses and named tuples now record with no configuration. Loading
  imports the class, allocates it with `object.__new__` and sets the attributes back
  (a slot through its descriptor, so frozen classes work); `__init__`, `__post_init__`,
  `__getstate__` and `__setstate__` are not used. Record method receivers are captured
  and rebuilt the same way.
- **Named tuples are stored by field name**, and diffs name the field
  (`$.output.ask`, not `$.output[1]`).
- **What cannot be rebuilt is refused**, at record time: functions, classes, modules,
  and C types holding state of their own (open files, locks, exceptions,
  compiled-extension objects), directly or as an attribute. There is no registration.
- A recorded attribute that no longer fits its class - a removed slot, an added
  dataclass or named-tuple field - fails loading with a message naming it; replay
  reports `not_loadable`.
- Schema version 3. Version 2 records still load and replay: an object's `"state"`
  and a `"namedtuple"`'s positional `"items"` are read as before. A call whose
  arguments hold an object or named tuple gets a new input hash, so re-recording it adds
  a record beside the old one.

### Added

- `types.MappingProxyType` and dicts with non-string keys are recorded (the latter as
  `[key, value]` pairs, keeping the keys' types), and a `mappingproxy` is compared key
  by key as its own kind.

### Removed

- `GenericObjectSerializer`, `GenericObjectComparator` and the registries' `fallback=`
  argument. The tree walker falls back to one object serializer when no registered
  serializer claims a value, and every registry loads the object tags with it; the
  comparison walker compares objects by their attributes after the registry.
  `register_serializer` / `register_comparator` place an entry behind the built-ins,
  still ahead of the object path. The tags `"object"` and `"namedtuple"` are reserved.
- `NamedTupleSerializer`.

## 0.3.0

Recording sessions now leave a record of the run itself: what it recorded, and under
which code, packages and data.

### Added

- `session(store, context=..., repos=..., packages=...)`: a context manager that lists
  every record made to `store` inside it, as `new` (written by the session) or
  `matched` (the same behavior was already recorded), and writes the list to
  `<root>/_runs/run-<run_id>.json` on exit. The manifest also holds the environment and
  the caller's JSON-safe `context` (a data version, a seed). It is written even when
  the block raises, marked `"completed": false`. Records, record ids and replay are
  unchanged.
- `environment(repos=..., packages=...)` and `Environment`. They describe the Python
  version and platform, the version and install source of each named distribution
  (with the commit, for a git install), and `HEAD` and a dirty flag for each named
  repository. The dirty flag counts only tracked files. An unknown entry is recorded as
  `None`, with a warning.
- `ReplayReport.save(path, environment=..., context=...)` writes a replay's results
  as JSON beside the environment it ran in.
- `regression_records.manifest.load_manifest` and `iter_manifest_paths`.

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
