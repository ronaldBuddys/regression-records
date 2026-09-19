# Comparison

`compare()` answers one question - *how* does this value differ from the one that was
recorded - and answers it completely.

```python
from regression_records import compare

result = compare(expected, actual)
if not result:
    print(result.summary())
```

Two properties shape everything below:

- **It collects every difference, never short-circuiting on the first.** A run that
  changed four columns should tell you about four columns, not send you round the loop
  four times. (Comparisons that would be *noise* after an earlier finding do stop - see
  [Short-circuiting](#short-circuiting).)
- **It never raises for a mismatch.** `compare()` is a pure function. Turning a failed
  comparison into a `RegressionMismatchError` is the decorator layer's policy, which is
  what keeps the comparison logic independent of recording.

## `ComparisonResult`

| Field / method | Meaning |
|---|---|
| `passed: bool` | True when no diffs were found. |
| `diffs: list[Diff]` | Every difference found, in traversal order. |
| `__bool__()` | Truthy iff `passed`, so `if not result:` reads naturally. |
| `summary(max_diffs=20) -> str` | A multi-line human summary; the message of `RegressionMismatchError`. |

Reachable after a decorated call as `wrapper.last_result`, whether or not the call
raised.

## `Diff`

| Field | Meaning |
|---|---|
| `path: str` | Where the difference is - see [Paths](#paths). |
| `kind: str` | What kind of difference - see below. |
| `expected: str` | A short **repr** of the recorded value. |
| `actual: str` | A short **repr** of the fresh value. |
| `message: str` | A sentence describing the difference. |
| `tolerance_used: ToleranceInfo \| None` | Set when a numeric tolerance decided the outcome. |

`expected` and `actual` are reprs, truncated to 200 characters - never the live objects.
A `Diff` outlives the comparison inside a `RegressionMismatchError`, and must not pin a
1GB array in memory to describe a difference in one of its cells.

### Diff kinds

| Kind | Meaning |
|---|---|
| `type_mismatch` | The two sides are different types (`list` vs `tuple`, array vs scalar). |
| `value_mismatch` | Same type, different value - beyond tolerance where tolerance applies. |
| `shape_mismatch` | Two arrays of different shape. |
| `dtype_mismatch` | Two arrays or numpy scalars of different dtype. |
| `missing_key` | Present in the recorded value, absent from the fresh one. Also used for set members. |
| `extra_key` | Present in the fresh value, absent from the recorded one. |
| `length_mismatch` | Sequences of different length, or dataframes of different height. |
| `schema_mismatch` | Dataframe columns, their dtypes, or their order differ. |
| `exception_mismatch` | One side raised where the other did not, or raised a different type. |

### `ToleranceInfo`

Attached to a numeric `Diff` so the report says what the value was measured against:
`rtol`, `atol`, and `max_abs_diff` - the largest finite absolute difference among the
mismatched elements, or `None` when it isn't measurable (infinities, complex dtypes).

## Paths

Every `Diff` carries a path naming exactly where it is. The grammar lives in
`regression_records.comparison.paths` and is **public** - `ToleranceConfig.path_overrides`
keys are matched against it.

| Form | Means |
|---|---|
| `$` | The value being compared. |
| `$.output` | A recorded call's return value. |
| `$.state_before` | A method receiver's state before the call. |
| `$.state_after` | ... and after it. |
| `$.key` | A dict entry, or an object attribute. |
| `$[0]` | A list or tuple element, by position. |
| `$['price']` | A dataframe column. |

They nest, so a real path reads like the access that would reach the value:
`$.state_after.frame['price']`, or `$.output.rows[3].weights`.

The three roots exist so one record's diffs stay distinguishable, and so each part can be
targeted by its own tolerance override.

## Tolerance

```python
from regression_records import ToleranceConfig

ToleranceConfig(
    rtol=1e-7,  # relative tolerance
    atol=0.0,  # absolute tolerance
    equal_nan=True,  # NaN in the same position on both sides is equal
    strict_dtype=True,  # a dtype change is a failure
    path_overrides={},  # per-subtree overrides
)
```

Two numbers are close when `abs(actual - expected) <= atol + rtol * abs(expected)` -
`numpy.isclose` semantics.

Infinities compare exactly: `inf` matches `inf` and nothing else, regardless of `rtol`.
NaN is governed by `equal_nan`, which defaults to True because a recorded NaN is usually
a meaningful "no value here" rather than an error.

`bool` is deliberately excluded from numeric comparison despite subclassing `int`: `True`
is not "1, within tolerance" - it is a different kind of answer. The same holds for a
boolean pandas column, though pandas itself counts one as numeric.

Integers *are* compared within tolerance, like floats - so with the default `rtol` a
count of 1,000,000,000 matches 1,000,000,050. Give counts and identifiers an exact
override (`ToleranceConfig(rtol=0, atol=0)`) where that matters. The exception is an
integer beyond 2**53, which a float cannot represent exactly: those are compared
exactly, whatever the tolerance.

### Why a dtype change is a failure

`strict_dtype=True` is the default, so `float64` becoming `float32` is reported even when
every value still agrees to within `rtol`.

The values agreeing today is not the point. A dtype change is a change in what the code
promises: different precision, a different overflow point, and different behavior on
inputs your recorded case does not happen to cover. It is a thing you want to have chosen
rather than to have drifted into. Set `strict_dtype=False` when you have chosen it; it
relaxes arrays, numpy scalars, Series and dataframe schemas alike (a dataframe's
columns must still have the same names).

### `path_overrides`

Tolerance is often not uniform. A price column may need to be exact while a
Monte-Carlo estimate needs 1%:

```python
ToleranceConfig(
    rtol=0.0,
    path_overrides={
        "$.output['estimate']": ToleranceConfig(rtol=1e-2),
        "$.state_after": ToleranceConfig(rtol=1e-9),
    },
)
```

An override applies to its path **and everything beneath it**. When several match, the
**most specific - the longest - prefix wins**, so a broad override on `$.output` can be
narrowed for one column beneath it.

An override replaces the config wholesale rather than merging field by field: the config
that applies at a path is the one you wrote there, not a blend of it with its ancestors.

Note that overrides are matched against the *reported* path, so they are only as stable
as your data's shape. A key rename changes the path.

## What gets compared how

The walker handles containers itself and delegates "inconvenient" leaves to a
`Comparator`, mirroring how serialization dispatches to its serializer registry.

**Dicts** are compared key by key, reporting `missing_key` and `extra_key` for each
difference in the key sets - both directions, so you see what appeared as well as what
vanished.

**Lists and tuples** are compared elementwise by position, and are **not**
interchangeable: a function that started returning a list where it used to return a tuple
has changed its contract. A **named tuple** is recorded with its class and is its own
kind: it matches the same named tuple class, and not a plain tuple. Any other `list` or
`tuple` subclass is recorded as the builtin, and compares as one.

**Sets** are compared by membership. There is no stable position to pair members up by,
so tolerance cannot meaningfully apply.

**Numpy arrays** check shape, then dtype, then values - numerically with tolerance for
numeric dtypes, exactly otherwise. A shape mismatch stops there; so does a dtype mismatch
between incomparable kinds (float vs str), where an elementwise comparison would be
meaningless.

**Numpy scalars** keep their dtype through recording, so `float32` vs `float64` is a
`dtype_mismatch`. When only one side is a numpy scalar - recorded as `np.float64`,
recomputed as a plain `float` - there are not two dtypes to compare and the values are
simply checked numerically.

**Dataframes** (polars and pandas) compare schema, then height, then column by column.
Numeric columns use tolerance; others compare exactly. Nulls surface as NaN, so
`equal_nan` governs whether two nulls in the same position count as equal.

**Custom objects** are compared attribute by attribute, from `__getstate__()` when the
class defines one and `__dict__` otherwise - the same source the generic serializer
records. Each attribute goes back through the top-level walker, so a nested array or
dataframe is compared by *its* comparator, with the right path and tolerance. Two objects
of different classes are a `type_mismatch` before any attribute is examined. An object
with no inspectable state falls back to plain equality, and says so in the message.

**Exceptions** compare on `exc_type` only. Messages routinely carry nondeterministic
detail - addresses, paths, timings - so a changed message is not a regression, while a
changed type is. An exception on one side and a value on the other is always a failing
`exception_mismatch`.

## Short-circuiting

The rule is: stop when continuing would produce noise rather than signal.

| After | Skipped | Because |
|---|---|---|
| A sequence length mismatch | Elementwise comparison | Positions no longer line up; every element after an insertion would be reported as changed. |
| A dataframe schema mismatch | Height and values | The columns being compared aren't the same columns. |
| A dataframe height mismatch | Column values | As above, for rows. |
| An array shape mismatch | Values | There is nothing to line up. |
| An object type mismatch | Attributes | Different classes; matching attribute names would be coincidence. |

Everything else keeps going and collects.

## Diff caps

Two caps keep a report readable when something changed globally.

**Within a column or array**, at most 20 positions are listed. The message always states
the true count: `"312 of 1000 element(s) differ beyond tolerance at index/indices: 0, 1,
2, ... (309 more)"`.

**Across columns**, once more than 20 columns have differed, the per-column diffs already
collected for the remainder are dropped and replaced by one summary diff. A frame where
every column shifted should say so once, not four hundred times.

Both default to 20 and are constructor arguments on the array and dataframe comparators,
for a registry you build yourself.

`ComparisonResult.summary()` applies its own separate cap of 20 lines to whatever it is
given; `result.diffs` always holds everything.

## Writing a comparator

A `Comparator` gives your own type useful diffs instead of plain equality:

```python
from regression_records import register_comparator
from regression_records.comparison.base import Diff


class TimeSeriesComparator:
    def can_handle(self, expected, actual):
        # Called with either side matching - the fresh value may be something else
        # entirely, which is itself the difference worth reporting.
        return isinstance(expected, TimeSeries) or isinstance(actual, TimeSeries)

    def compare(self, expected, actual, path, tolerance, compare_fn, diffs):
        if expected.freq != actual.freq:
            diffs.append(
                Diff(
                    path=path,
                    kind="schema_mismatch",
                    expected=str(expected.freq),
                    actual=str(actual.freq),
                    message=f"frequency {actual.freq} != expected {expected.freq}",
                )
            )
            return
        compare_fn(expected.values, actual.values, path)  # recurse for the array


register_comparator(TimeSeriesComparator())
```

Three things to know:

- **`can_handle` is asked about both sides** and dispatches if *either* matches. Handling
  "only one side is my type" is part of the job; `comparison.base.require_both` does it
  for you.
- **Append to `diffs`; do not return anything.** `compare_fn(expected, actual, path)`
  recurses into the top-level walker and appends to the same list, so a comparator that
  both reports its own findings and recurses cannot double-count them.
- **`tolerance` arrives already resolved** for this path, with any override applied.

`register_comparator` places it ahead of the generic object fallback but behind the
built-in comparators. Pass `comparator_registry=` on a decorator instead when the change
should be scoped to one decoration rather than global.
