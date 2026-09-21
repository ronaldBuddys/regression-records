# Schema version 1 store

Recorded by regression-records 0.1.0 (schema version 1, `main` at 711dd4b) and checked in
unchanged, apart from the traceback text, which was shortened to drop local paths. Do not
regenerate it: `tests/test_record_format.py` uses it to check that v1 records still load
and compare. It holds bare `NaN` / `Infinity` tokens and a bare exception name
(`"ValueError"`).

Source of the recorded functions (module `v1_fixture_source`):

```python
import numpy as np

from regression_records import record_func

STORE = "tests/fixtures/store_v1"


@record_func(STORE)
def non_finite(x):
    return {"nan": float("nan"), "inf": float("inf"), "ninf": -float("inf"), "x": x}


@record_func(STORE)
def numpy_non_finite(x):
    return [np.float64("nan"), np.float64("inf"), np.float32("-inf"), x]


@record_func(STORE)
def raises(x):
    raise ValueError(f"bad {x}")


non_finite(1.5)
numpy_non_finite(2.5)
raises(3)  # raises ValueError, recorded
```
