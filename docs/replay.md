# Replay

Recording captures what a function did. `replay()` checks that it still does,
possibly much later, after the code or the environment has changed:

```python
from regression_records import ToleranceConfig, replay

report = replay("tests/regression_data", tolerance=ToleranceConfig(rtol=1e-9))
assert report.passed, report.summary()
```

For each record in the store, replay:

1. imports the function by its recorded module and qualname;
2. loads the recorded arguments;
3. calls the *original* function, the one beneath the decorator;
4. reconciles what it returned or raised with the record, exactly as the decorator's
   validation does (the same comparators, tolerances and diff paths).

Every record is replayed on its own. Nothing is looked up by input hash, so two records
of the same call that disagree give one pass and one failure. Deciding what such a
failure means (an intended change, a bug fix, a regression) is left to you. Each case
carries its record's `input_hash` and `created_at`, so records of one call can be
grouped.

## Replay only reads

Replay never writes to a store. It calls the function beneath the decorator, and while
it runs no decorated call anywhere in the process records. That holds for a decorated
function the replayed one calls, including from a worker thread, and for a subprocess
it starts, since `REGRESSION_RECORDS` is set to `0` for the duration.

## The report

`replay()` returns a `ReplayReport`: one `ReplayCase` per record, in record-path order.
`report.passed` is true when every case passed, and `report.summary()` lists the counts
and each failure. `report.to_dict()` is JSON-safe, for keeping results over time.

| Status | Meaning |
| --- | --- |
| `passed` | The call behaved as recorded, within tolerance. |
| `differs` | The result, or the exception raised, differs from the record. This includes a call that now returns where the record says it raised. |
| `raised` | The record says the call returned, but on replay it raised. Usually the call could not run outside its original context, or its signature changed. |
| `not_importable` | The recorded module or function no longer exists under its recorded name. |
| `not_replayable` | The name resolves, but not to the recorded function. It is no longer decorated, or it is now another function's recorder. Method records are not replayed yet. |
| `not_loadable` | The record file, or a value it holds, cannot be read. |

**Code drift is a column, not a failure.** Replay runs current code on old inputs, so
the source having changed since the record was made is expected. `case.code_drift`
says whether a pass means "behavior held across a change" or only "nothing changed".

Each case keeps at most `max_diffs` diffs (default 100); `case.diffs_truncated` counts
the rest, so a report over a large store stays small.

## One pytest test per record

`replay_tests()` turns a store into a pytest suite, so a failure names its case and
`pytest -k` can pick records out:

```python
# tests/test_regressions.py
from regression_records import ToleranceConfig, replay_tests

test_regressions = replay_tests(
    "tests/regression_data", tolerance=ToleranceConfig(rtol=1e-9)
)
```

It takes the same arguments as `replay()`. Bind the result to a name starting with
`test_`, or pytest will not collect it. Each record becomes one parametrized case, with
an id of the form `module:qualname@<record id>`. A case fails unless it passed, and
`not_loadable`, `not_importable` and `not_replayable` fail too rather than being
skipped. A case that passed although its function's source has changed since it was
recorded warns with `CodeDriftWarning`. A store with no records gives pytest's
"empty parameter set" skip. A store path that does not exist fails at collection.

## Choosing records and tolerances

```python
replay(
    store,
    tolerance=ToleranceConfig(rtol=1e-9),
    tolerance_overrides={"pricing.*:price_*": ToleranceConfig(rtol=1e-6)},
    select="pricing.*",
)
```

- `tolerance_overrides` keys are globs on `"module:qualname"`. The first match wins,
  and everything else gets `tolerance`. Within one function, `path_overrides` on a
  `ToleranceConfig` still target individual values.
- `select` is a glob on `"module:qualname"`, or a predicate taking the
  `RegressionRecord`. A record that cannot be read is matched by the name its location
  implies, and a predicate always keeps it, so corruption is never silently skipped.

## What replay cannot see

- **A call is replayed from its recorded inputs alone.** A function that reads a clock,
  a database, an environment variable or any other global state sees today's value,
  not the recorded one. Expect `raised` or `differs` from such a function. That is a
  limit of replay, not necessarily a regression.
- **Side effects happen again.** A replayed function that writes files or sends
  requests does so again, as it would in a unit test.
- **An argument must load back as the value that was recorded.** Where a type does not
  round-trip exactly (see [Limitations](limitations.md#values-that-record-but-lose-something)),
  the replayed call receives something slightly different, which can surface as a
  failure.
- **The function must be importable by name.** A lambda, a closure, or a function
  defined in a script run as `__main__` cannot be found again.
- **A store path that does not exist is an error**, not an empty pass. An existing but
  empty store replays nothing and passes.
