# regression-records

Record what your functions do, then find out when they stop doing it.

Decorate a function, point it at a directory, and run your code with recording switched
on. Each call is written to disk as a **regression record**: its arguments, and what it
returned or raised. Later - after a refactor, a dependency upgrade, on another machine -
**replay** reruns every recorded call against the current code and reports each
difference with the exact path that changed:

```
replayed 12 record(s) from tests/regression_data: 11 passed, 1 differs
differs: myapp.pricing:score record 0f3a9c2e (myapp/pricing/score/records/rec-...)
  comparison failed with 1 diff(s):
    at "$.output['price']" (value_mismatch): 3 of 1000 value(s) differ beyond
    tolerance at row(s): 41, 88, 512
```

This is characterization testing, or "golden master" testing: useful when you are about
to refactor code whose behavior you trust but have not written tests for, when you want
to know that a dependency upgrade changed nothing, or when the interesting output is a
dataframe nobody wants to write assertions about by hand.

---

## Install

Python 3.11 or later. The package is not on PyPI; install it from git:

```bash
uv add "regression-records @ git+https://github.com/ronaldBuddys/regression-records"
uv add "regression-records[polars] @ git+https://github.com/ronaldBuddys/regression-records"
uv add "regression-records[pandas] @ git+https://github.com/ronaldBuddys/regression-records"
```

The bare install records numpy values; `[polars]` adds polars DataFrame/Series,
`[pandas]` adds pandas DataFrame/Series (with pyarrow, which writes them).

numpy is a hard dependency. polars and pandas are optional: without them the package
imports and records normally, and their serializers are simply not registered - you
cannot pass a DataFrame you have no library to construct.

## Quickstart: record, then replay

Decorate the function and say where its records live:

```python
# myapp/pricing.py
from regression_records import record_func


@record_func("tests/regression_data")
def score(frame, weights):
    return (frame * weights).sum()
```

**Record** by running anything that calls it with the switch on - a script, or your
existing tests:

```bash
REGRESSION_RECORDS=1 python run_pipeline.py
REGRESSION_RECORDS=1 pytest
```

`tests/regression_data/` fills with JSON records plus any arrays or dataframes they
reference. Commit that directory. Without the switch the decorator records nothing and
costs one check per call, so it can stay on production code.

**Replay** in a test module, which pytest turns into one test per record, named after
the function and the record:

```python
# tests/test_regressions.py
from regression_records import replay_tests

test_regressions = replay_tests("tests/regression_data")
```

or from Python, for a report you can inspect or keep:

```python
from regression_records import ToleranceConfig, replay

report = replay("tests/regression_data", tolerance=ToleranceConfig(rtol=1e-9))
assert report.passed, report.summary()
```

Replay imports each function by its recorded module and name, calls the function
beneath the decorator on the recorded arguments, and compares what it returns (or
raises) with the record. It never writes: while it runs, nothing records, whatever the
switch says. [docs/replay.md](docs/replay.md) covers the report, tolerances and what
replay cannot see.

The wrapper is transparent: arguments arrive untouched, the return value passes through,
and an exception the function raises is always re-raised. Nothing is compared while a
decorated function runs; checking is replay's job alone.

## The switch

Recording is off unless the environment variable `REGRESSION_RECORDS` says otherwise.
It is read on every call, so a test can flip it, and being an environment variable it
reaches subprocesses and CI jobs without code changes.

| `REGRESSION_RECORDS` | Recording |
|---|---|
| `1`, `true`, `yes`, `on` (any case) | on |
| unset, empty, `0`, `false`, `no`, `off` | off |
| anything else | `ValueError` on the next decorated call - a typo must not silently leave recording off |

From Python, `regression_records.enable()` and `disable()` set and remove it, and
`with recording():` switches it on for a block and restores it afterwards. The switch is
process-wide. `enabled=False` on a decorator turns that one function off whatever the
environment says.

## Quickstart: a method

A method's behavior is as much about the object it acts on as about its arguments, so
`record_method` snapshots `self` **before** and **after** the call, and records both:

```python
from regression_records import record_method


class Portfolio:
    def __init__(self, holdings):
        self.holdings = holdings
        self.history = []

    @record_method("tests/regression_data")
    def rebalance(self, target):
        self.history.append(dict(self.holdings))
        self.holdings = _rebalanced(self.holdings, target)
        return self.holdings
```

With both snapshots recorded, a refactor that makes `rebalance` return the right answer
but forget to append to `history` shows up as a difference in `$.state_after` - the
regression `record_func` structurally cannot see.

Replay rebuilds `self` in its recorded state before the call, calls the method on the
recorded arguments, and compares both the return value and the state after. Rebuilding
from the whole recorded state works as unpickling does - `cls.__new__(cls)`, then
`__setstate__` or `__dict__.update` - so `__init__` is not called, and anything it sets
up without storing it as an attribute is not there on replay.

Two things about `self` are worth knowing:

- **It is not part of the recorded arguments, or of the input hash.** The object is
  described by its captured state instead. Hashing the instance would make every
  instance - and, since most objects hash by identity, every run - a separate call.
- **The state still tells calls apart.** Two instances in different states calling with
  the same arguments give two records, filed together under the same input hash.

State that cannot or should not be recorded - a database connection, an open file, a
cache - is excluded by name, or, from the other direction, `state_attrs=("holdings",)`
records only what you name. Either way the record holds only part of the object, which
is not enough to rebuild it, so `factory=` is required: a zero-argument callable
returning an instance, on which replay sets the recorded attributes.

```python
def new_quotes():
    return Quotes(connect())


class Quotes:
    @record_method(store, exclude_attrs=("conn", "_cache"), factory=new_quotes)
    def query(self, symbol): ...
```

## Re-recording after an intended change

Sooner or later a failing replay is *correct*: you changed the behavior on purpose.
Records are never replaced or deleted automatically, so accepting the change is two
steps:

1. **Re-record** with the switch on (`REGRESSION_RECORDS=1 pytest`). The new behavior is
   written as a new record beside the old one.
2. **Delete the old record file** - replay now fails on it, and the failure names its
   path - then collect the data files nothing references any more:

   ```python
   RecordStore("tests/regression_data").prune(dry_run=False)
   ```

Nothing is replaced for you because the new behavior is not always the right one: an
automatic overwrite would lose a correct record to a buggy change. The deletion, and the
new record, show up in `git diff` on the store, where a reviewer can look at them.

Recording behavior identical to an earlier recording writes nothing, so re-recording a
whole suite only adds records for what actually changed.

## Configuration

Both decorators take the same options (`record_method` adds `state_attrs`,
`exclude_attrs` and `factory`).

| Option | Default | What it does |
|---|---|---|
| `store` | *required* | A `RecordStore`, or a path to wrap in one. There is no default location - recording somewhere implicit would be worse than being asked. |
| `record_exceptions` | `True` | Whether a call that raises is recorded, with the exception in place of a return value. |
| `tags` | `None` | Free-form metadata stored on each record this decoration writes. |
| `serializer_registry` | module default | An injected registry, for recording your own types. |
| `enabled` | `True` | `False` never records, whatever `REGRESSION_RECORDS` says. |
| `state_attrs` | `None` | `record_method` only: the attributes to record, or `None` for all of them. |
| `exclude_attrs` | `()` | `record_method` only: attributes to leave out. |
| `factory` | `None` | `record_method` only: a zero-argument callable that builds the receiver for replay. Required with `state_attrs` or `exclude_attrs`. |

How results are compared - tolerances, per-function overrides, custom comparators - is
set on `replay()`, not on the decorator: recording has no use for it. See
[docs/replay.md](docs/replay.md) and [docs/comparison.md](docs/comparison.md).

## Recording your own types

Anything with a `__dict__` or a `__getstate__()` is recorded attribute by attribute for
free. When that is not what you want - a type with a compact binary form, or one whose
attributes are noise - write a `Serializer`:

```python
from regression_records import Serializer, register_serializer


class TimeSeriesSerializer:
    type_tag = "myapp.TimeSeries"  # written into the record; used to load it back

    def can_handle(self, obj):
        return isinstance(obj, TimeSeries)

    def store(self, obj, ctx):
        filename = f"ts-{obj.checksum()}.bin"
        ctx.write_if_absent(filename, lambda path: path.write_bytes(obj.to_bytes()))
        return {"path": ctx.ref(filename), "length": len(obj)}

    def load(self, blob, ctx):
        return TimeSeries.from_bytes(ctx.resolve(blob["path"]).read_bytes())


register_serializer(TimeSeriesSerializer())
```

`register_serializer` inserts it ahead of the generic object fallback but behind the
built-in serializers. A matching `Comparator` (via `register_comparator`) gives the type
useful diffs rather than plain equality; see
[docs/comparison.md](docs/comparison.md#writing-a-comparator).

`register_serializer` changes `default_registry`, which every decoration in the process
shares, and which replay uses to load records back. Replay imports each recorded
function's module before loading its values, so a serializer registered at import time
in that module is in place when it is needed.

To scope the change to one decoration, copy the default registry and inject the copy -
appropriate in a library, or in a single test:

```python
from regression_records.serialization import default_registry

registry = default_registry.copy()
registry.register(TimeSeriesSerializer())


@record_func(store, serializer_registry=registry)
def load_series(path): ...
```

Replay the store through a `RecordStore` carrying the same registry,
`replay(RecordStore("tests/regression_data", serializer_registry=registry))`, so the
values load back. `ComparatorRegistry.copy()` works the same way, injected with
`replay(..., comparator_registry=...)`.

## Exceptions

Everything raised by this library subclasses `RegressionRecordsError`.

| Exception | Raised when |
|---|---|
| `SerializationError` | A value cannot be recorded - an object-dtype array, a non-string dict key, a class defined in `__main__`, state that cannot be deep-copied. |
| `DeserializationError` | A stored value cannot be reconstructed. |
| `CorruptRecordError` | *(a `DeserializationError`)* A record's JSON, or a file it references, is missing or malformed. |
| `IdentityResolutionError` | A recorded class can no longer be imported by name. |
| `CodeDriftWarning` | *(a warning, not an error)* Warned by `replay_tests` when a record passes but its function's source has changed since it was recorded. Not a failure - it tells you whether a pass means "behavior held across a change" or just "nothing changed". `replay()` reports the same thing as `case.code_drift`. |

A decoration that could never be recorded or replayed raises a plain `TypeError` at
import time: a generator or async function, a lambda, a closure, a `functools.partial`,
anything defined in `__main__`, and `state_attrs` or `exclude_attrs` without `factory=`.

A regression is not an exception: it is a failed case in the `ReplayReport`, or a
failed test under `replay_tests`.

A `SerializationError` is deliberately *not* swallowed. A store that quietly records
nothing passes vacuously; leaving the switch off, or `enabled=False`, is the way not to
record.

Two behaviors worth stating outright:

- **The wrapped function's own exception always wins.** It is re-raised unchanged, and
  recorded as the call's outcome unless `record_exceptions=False`.
- **`KeyboardInterrupt` and `SystemExit` propagate unrecorded.** They mean the process is
  going away, not that the function misbehaved.

## Further reading

The documentation is a Sphinx site, built from this repository:

```bash
uv sync --all-extras
uv run make -C docs html && open docs/_build/html/index.html
```

- [docs/getting_started.rst](docs/getting_started.rst) - install, recording a function
  and a method, replaying from pytest, and re-recording after a change.
- [examples/](examples/) - fourteen runnable scripts, rendered as the docs' gallery: a
  tour from a first recording to your own serializers, then a realistic pricing
  pipeline, the gotchas, and `replay_tests`. `python examples/01_first_record.py`.
- [docs/replay.md](docs/replay.md) - what replay does, the report and its statuses,
  tolerances per function, and what replay cannot see.
- [docs/persistence.md](docs/persistence.md) - what the store looks like on disk, how
  files are named and deduplicated, what survives moving it, and what to trust.
- [docs/comparison.md](docs/comparison.md) - the `Diff` and `ComparisonResult` reference,
  the path grammar, and tolerance configuration.
- [docs/limitations.md](docs/limitations.md) - everything it does not do, starting with
  the ways a replay can pass without testing anything.
- [docs/design_notes.md](docs/design_notes.md) - decisions that are not visible from
  the API.

## Known limitations

The short list; [docs/limitations.md](docs/limitations.md) is the full one.

- **polars `LazyFrame` is refused**, not auto-collected: collection can be expensive or
  side-effecting, so `.collect()` is yours to call.
- **`dtype=object` numpy arrays are refused** - they are not portable and require
  unpickling to load. So are `ndarray` subclasses such as `MaskedArray`: only the data
  buffer would be saved, silently dropping the mask. Record the parts as plain arrays.
- **An object that refers back to itself cannot be recorded** - a record is a tree.
  Leave a parent/child back-reference out with `exclude_attrs`.
- **Non-string dict keys are refused.** JSON has no other kind; convert before recording.
- **Classes defined in `__main__` or inside a function cannot be recorded** - they
  cannot be re-imported by name from a fresh process later.
- **A pandas frame Arrow cannot represent** (mixed dtype object columns) **is refused**
  unless you opt in to pickling that one file, with
  `PandasDataFrameSerializer(allow_pickle=True)`; a pickle is never read without the
  same opt-in. `DatetimeIndex.freq` is not preserved through parquet.
- **Loading a store is not safe on a store you do not trust.** A recorded custom object
  names its class, and loading imports that module. Review changes to a store as you
  review the code beside it.
- **Moving a function between modules is not auto-resolved** - it is ambiguous in
  general. `RecordStore.rename_identity(old, new)` moves the history by hand.
- **Replay runs from recorded inputs alone.** A function reading a clock, a database,
  an environment variable or a global sees today's value, and its side effects happen
  again. It must be importable by name: lambdas, closures, `functools.partial`
  objects and functions defined in `__main__` are refused when decorated.
- **A method's receiver is rebuilt without `__init__`** from its whole recorded state,
  or by `factory=`, which `state_attrs` and `exclude_attrs` require.
- **Several processes may record at once** - identical recordings converge on one file,
  changed ones add records - but do not run `RecordStore.prune` while anything records.
- **Record files are ordinary files**, readable according to your umask, holding every
  recorded argument in full. Do not record calls whose arguments carry secrets.
- **There is no CLI** for browsing or pruning a store; it is JSON in directories, and
  ordinary tools work on it.

## Development

```bash
uv sync                  # the dev group brings pytest, ruff, polars, pandas, pyarrow
uv run pytest
uv run ruff check . --fix
uv run ruff format .
```

CI runs exactly these, on Python 3.11 to 3.14.

The suite runs with polars and pandas installed, so it cannot check that a *bare*
install behaves. That is a separate, manual check:

```bash
uv build
uv venv /tmp/smoke && uv pip install --python /tmp/smoke dist/regression_records-*.whl
/tmp/smoke/bin/python -c "import regression_records; print(regression_records.__version__)"
```

The bare install must import cleanly and record numpy values; the polars and pandas
serializers are simply absent rather than failing at import time.

## License

MIT - see [LICENSE](LICENSE).
