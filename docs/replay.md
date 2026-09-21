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
2. loads the recorded arguments, and for a method rebuilds the object it was called on
   (see [Methods](#methods));
3. calls the *original* function, the one beneath the decorator;
4. compares what it returned or raised with the record, and for a method the object's
   state afterwards, using the comparators and tolerances described in
   [Comparison](comparison.md), and reports each difference by path
   (`$.output['price']`, `$.state_after.history`).

Replay is the only place a record is checked. The decorators only record; nothing is
compared while a decorated function runs.

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

## Methods

A `record_method` record holds the receiver's state before and after the call. Replay
imports the method's class by the qualname's leading part (`Ledger` for
`Ledger.apply`), rebuilds an instance in the state before, calls the original method on
it with the recorded arguments, and reconciles two things with the record:

- what the method returned or raised, as for a function (`$.output`);
- the instance's state afterwards, captured the same way it was when recording
  (`$.state_after...`).

`state_before` is the call's input and is not compared. A method that still returns
the right value but leaves its object in a different state is reported as `differs`,
with the diff under `$.state_after`.

How the instance is rebuilt depends on what was recorded:

- **The whole state** (no `state_attrs` or `exclude_attrs`): `cls.__new__(cls)`, then
  the class's own `__setstate__(state)` if it defines one, else
  `instance.__dict__.update(state)`. This is what unpickling does. **`__init__` is not
  called**, so anything it establishes without storing it as an attribute - registering
  the object somewhere, opening a resource, validating an invariant - is not there on
  replay. A class with a `__getstate__` / `__setstate__` pair is recorded and rebuilt
  through them.
- **Part of the state** (`state_attrs` or `exclude_attrs`): a record like that cannot
  rebuild the object on its own, so the decorator requires `factory=`, a zero-argument
  callable returning an instance. Replay calls it, then sets each recorded attribute
  on the result with `setattr`. Leaving `factory=` out is a `TypeError` when the method
  is decorated.

```python
def new_pricer() -> Pricer:
    return Pricer(connect())


class Pricer:
    @record_method(STORE, exclude_attrs=("connection",), factory=new_pricer)
    def value(self, symbol: str, quantity: int) -> float: ...
```

The factory is called, not imported, so a lambda is fine. Recording is suppressed
while it runs, as it is for the replayed call.

**The current decorator decides.** A record does not store `state_attrs`,
`exclude_attrs` or `factory`; replay uses those of the decorator on the method now, and
records keep the format they had. So a `factory=` changed after recording applies to
the old records too. The state after is captured with the current `state_attrs` and
`exclude_attrs` but compared with the recorded one, so changing either makes old records
differ on the attributes that moved in or out (a `missing_key` diff under
`$.state_after` for an attribute now left out): re-record them.

A method case is `not_replayable`, with the reason in `case.error`, when:

- the factory returns something that is not an instance of the method's class;
- rebuilding the instance raises (`__new__`, `__setstate__` or the factory);
- the record is of a method but the name is now a recorded function, or the other way
  round.

## The report

`replay()` returns a `ReplayReport`: one `ReplayCase` per record, in record-path order.
`report.passed` is true when every case passed, and `report.summary()` lists the counts
and each failure. `report.to_dict()` is JSON-safe, for keeping results over time.

| Status | Meaning |
| --- | --- |
| `passed` | The call behaved as recorded, within tolerance. |
| `differs` | The result, the state after (for a method), or the exception raised differs from the record. This includes a call that now returns where the record says it raised. |
| `raised` | The record says the call returned, but on replay it raised. Usually the call could not run outside its original context, or its signature changed. |
| `not_importable` | The recorded module or function no longer exists under its recorded name. |
| `not_replayable` | The name resolves, but not to a recorded function replay can call. It is no longer decorated, it is now another function's recorder, it changed between function and method, or a method's receiver could not be rebuilt. |
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

The recorded functions are imported by name in the test process, so their package must
be importable there, as it is for your other tests.

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
- **The function must be importable by name.** A lambda, a function or class defined
  inside another function, a `functools.partial`, and anything defined in the script
  being run (`__main__`) are refused with a `TypeError` when decorated, since replay
  could never find them again. Records of such functions made by earlier versions
  replay as `not_importable`.
- **A method's object is rebuilt from its record, without `__init__`.** State the
  object keeps outside its own attributes - a module-level registry, a connection it
  opened - is not rebuilt. Use `factory=` when the recorded state is not enough.
- **A store path that does not exist is an error** (`FileNotFoundError`, at collection
  under `replay_tests`), not an empty pass. An existing but empty store replays nothing:
  `replay()` passes with zero cases, and `replay_tests` gives pytest's "empty parameter
  set" skip. A function with no records is simply not checked.
