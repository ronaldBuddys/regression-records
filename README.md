# regression-records

Record what your functions do, then find out when they stop doing it.

Decorate a function, point it at a directory, and run your code. The first time each
distinct set of arguments is seen, the call is written to disk as a **regression
record**. Every later run of that same call is compared against the recording, and a
difference is reported with the exact path that changed:

```
RegressionMismatchError: comparison failed with 1 diff(s):
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

The bare install records numpy values; `[polars]` adds polars DataFrames, `[pandas]`
adds pandas DataFrame/Series (with pyarrow, which writes them).

numpy is a hard dependency. polars and pandas are optional: without them the package
imports and records normally, and their serializers are simply not registered - you
cannot pass a DataFrame you have no library to construct.

## Quickstart: a function

```python
from regression_records import record_func


@record_func("tests/regression_data")
def score(frame, weights):
    return (frame * weights).sum()
```

Run it once and `tests/regression_data/` fills with JSON records plus any arrays or
dataframes they reference. Commit that directory. From then on, every run of `score`
with those same arguments is checked against what it produced before.

The wrapper is transparent: arguments arrive untouched, the return value passes through,
and an exception the function raises is always re-raised. Only a *mismatch* changes
control flow, and only when you asked it to.

In CI, where you want validation and never recording:

```python
@record_func("tests/regression_data", mode=RunMode.VALIDATE_ONLY)
def score(frame, weights): ...
```

## Quickstart: a method

A method's behavior is as much about the object it acts on as about its arguments, so
`record_method` snapshots `self` **before** and **after** the call, and compares both:

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

If a refactor makes `rebalance` return the right answer but forget to append to
`history`, that is caught - which is the regression `record_func` structurally cannot
see.

Two things about `self` are worth knowing:

- **It is not part of the recorded arguments, or of the lookup key.** The object is
  described by its captured state instead. Hashing the instance would make every
  instance - and, since most objects hash by identity, every run - a separate case.
- **The consequence:** two instances in *different* states calling with the *same*
  arguments are one case, not two. The state difference is reported as a mismatch. If
  the starting state is what distinguishes two calls, pass it as an argument.

State that cannot or should not be recorded - a database connection, an open file, a
cache - is excluded by name:

```python
@record_method(store, exclude_attrs=("conn", "_cache"))
def query(self, symbol): ...
```

or, from the other direction, `state_attrs=("holdings",)` records only what you name.

## Accepting an intentional change

Sooner or later a mismatch is *correct*: you changed the behavior on purpose. Re-run
with `OverwritePolicy.IF_MISMATCH` and the recorded baseline is replaced with the new
result:

```python
@record_func(store, overwrite=OverwritePolicy.IF_MISMATCH)
def score(frame, weights): ...
```

Rewriting the baseline *is* the act of accepting, so this does not raise. The diffs are
still available on `score.last_result` if you want to see what you accepted, and the
change shows up in `git diff` on the record files, where a reviewer can look at it.

Then take the flag back off. `IF_MISMATCH` left in place permanently is a suite that
agrees with whatever the code currently does.

## Configuration

Both decorators take the same options (`record_method` adds `state_attrs` and
`exclude_attrs`).

| Option | Default | What it does |
|---|---|---|
| `store` | *required* | A `RecordStore`, or a path to wrap in one. There is no default location - recording somewhere implicit would be worse than being asked. |
| `mode` | `RunMode.BOTH` | `RECORD_ONLY`, `VALIDATE_ONLY`, or `BOTH` (validate a known case, record an unknown one). |
| `tolerance` | `ToleranceConfig()` | Numeric comparison tolerances. See [docs/comparison.md](docs/comparison.md). |
| `on_new_case` | *mode-appropriate* | What to do when no record matches: `RECORD`, `ERROR`, or `IGNORE`. Unset means record it, or - under `VALIDATE_ONLY` - raise. |
| `overwrite` | `OverwritePolicy.NEVER` | When an existing record may be replaced: `NEVER`, `IF_MISMATCH`, `ALWAYS`. |
| `on_mismatch` | `OnMismatch.RAISE` | `RAISE`, or `RETURN` to inspect `wrapper.last_result` instead. |
| `enabled` | `True` | `False` makes the wrapper a pure pass-through with no store access. |
| `record_exceptions` | `True` | Whether a raised exception is recorded, and compared, in place of a return value. |
| `name` | `None` | Overrides the qualname used to identify this function on disk. |
| `tags` | `None` | Free-form metadata stored on each record this decoration writes. |
| `serializer_registry` / `comparator_registry` | module defaults | Injected registries, for recording your own types. |

`mode` and `overwrite` answer different questions and compose freely: **`mode` decides
whether a *new* case may be created; `overwrite` decides whether an *existing* record
may be replaced.** So `VALIDATE_ONLY` with `IF_MISMATCH` is the ordinary
accept-a-change workflow, and `VALIDATE_ONLY` with `ALWAYS` re-baselines every case you
already have without inventing any new ones.

The one combination that is genuinely contradictory - `VALIDATE_ONLY` with an explicit
`on_new_case=RECORD` - raises `ValueError` when the decorator is applied, rather than
weeks later on the first call that happens to reach novel inputs.

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
shares. To scope the change to one decoration, copy the default registry and inject the
copy - appropriate in a library, or in a single test:

```python
from regression_records.serialization import default_registry

registry = default_registry.copy()
registry.register(TimeSeriesSerializer())


@record_func(store, serializer_registry=registry)
def load_series(path): ...
```

`ComparatorRegistry.copy()` and `comparator_registry=` work the same way.

## Exceptions

Everything raised by this library subclasses `RegressionRecordsError`.

| Exception | Raised when |
|---|---|
| `SerializationError` | A value cannot be recorded - an object-dtype array, a non-string dict key, a class defined in `__main__`, state that cannot be deep-copied. |
| `DeserializationError` | A stored value cannot be reconstructed. |
| `CorruptRecordError` | *(a `DeserializationError`)* A record's JSON, or a file it references, is missing or malformed. |
| `RecordNotFoundError` | Validating a function that has never been recorded at all. |
| `UnmatchedInvocationError` | No record matches these inputs, and `on_new_case` does not permit creating one. |
| `RegressionMismatchError` | A comparison against history failed. Carries the full `ComparisonResult` on `.result`. |
| `IdentityResolutionError` | A recorded class can no longer be imported by name. |
| `CodeDriftWarning` | *(a warning, not an error)* Validating against a record captured from different source code. Not a failure - it tells you whether a pass means "behavior held across a change" or just "nothing changed". |

A `SerializationError` is deliberately *not* swallowed. A regression suite that quietly
records nothing passes vacuously; `enabled=False` is the way to turn recording off.

Two behaviors worth stating outright:

- **The wrapped function's own exception always wins.** If a call raises *and* mismatches
  its record, the real exception is what reaches you - the mismatch is on `last_result`.
  A suite still fails, because the exception propagates.
- **`KeyboardInterrupt` and `SystemExit` propagate unrecorded.** They mean the process is
  going away, not that the function misbehaved.

## Further reading

- [examples/](examples/) - eleven runnable notebooks, from a first recording to writing
  your own serializers. `python examples/01_first_record.py`.
- [docs/persistence.md](docs/persistence.md) - what the store looks like on disk, how
  files are named and deduplicated, and what is guaranteed to survive moving it.
- [docs/comparison.md](docs/comparison.md) - the `Diff` and `ComparisonResult` reference,
  the path grammar, and tolerance configuration.

## Known limitations

- **polars `LazyFrame` is refused**, not auto-collected: collection can be expensive or
  side-effecting, so `.collect()` is yours to call.
- **`dtype=object` numpy arrays are refused** - they are not portable and require
  unpickling to load.
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
- **Writes are not concurrency-safe.** One process at a time per store.
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
