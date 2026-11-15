Getting started
===============

Install
-------

Python 3.11 or later. The package is not on PyPI; add it to a ``uv`` project straight
from GitHub, with the extra for whichever dataframe library your code uses:

.. code-block:: bash

   uv add "regression-records @ git+https://github.com/ronaldBuddys/regression-records"
   uv add "regression-records[polars] @ git+https://github.com/ronaldBuddys/regression-records"
   uv add "regression-records[pandas] @ git+https://github.com/ronaldBuddys/regression-records"

Or work from a local checkout, with everything needed to run the examples and build
these docs:

.. code-block:: bash

   git clone https://github.com/ronaldBuddys/regression-records
   cd regression-records
   uv sync --all-extras

numpy is the only hard dependency. Without polars or pandas the package imports and
records normally; their serializers are simply not registered.

Record a function
-----------------

Decorate it and say where its records live. There is no default location, because
recording somewhere implicit would be worse than being asked.

.. code-block:: python

   # myapp/rates.py
   import numpy as np
   from regression_records import record_func

   @record_func("tests/regression_data")
   def discount_factors(rate: float, years: np.ndarray) -> np.ndarray:
       return np.exp(-rate * years)

The function lives in a module (``myapp/rates.py``) because replay later imports it by
name. Decorating one that has no importable name - a lambda, a function defined inside
another, or one in the script being run (``__main__``) - raises ``TypeError``.

Nothing is recorded yet: recording is off until the environment variable
``REGRESSION_RECORDS`` switches it on. Run whatever calls the function - a script, or
your existing tests - with the switch set:

.. code-block:: bash

   REGRESSION_RECORDS=1 python run_pipeline.py
   REGRESSION_RECORDS=1 pytest

Each call writes a record: the arguments, and what the call returned or raised (so that
"raises ``ValueError``" is regression-tested too). Commit the store directory. A call
repeated with the same arguments is recorded once, and recording behaviour identical to
an earlier run writes nothing.

From Python, :func:`~regression_records.enable` and :func:`~regression_records.disable`
set and remove the variable, and :func:`~regression_records.recording` switches it on
for a ``with`` block:

.. code-block:: python

   from regression_records import recording

   with recording():
       discount_factors(0.03, np.array([1.0, 2.0, 5.0]))   # recorded

The function itself is untouched: it receives its original arguments, its return value
passes through, and an exception it raises is always re-raised. With the switch off the
decorator costs one check per call, so it can stay on production code.

Record a method
---------------

A method's behaviour is as much about the object as about its arguments, so
:func:`~regression_records.record_method` also snapshots ``self`` before and after the
call. A method that returns the right answer by way of the wrong mutation shows up as a
difference in its state.

.. code-block:: python

   from regression_records import record_method

   class Portfolio:
       def __init__(self):
           self.positions = {}

       @record_method("tests/regression_data")
       def trade(self, symbol: str, quantity: int) -> int:
           self.positions[symbol] = self.positions.get(symbol, 0) + quantity
           return self.positions[symbol]

Replay rebuilds ``self`` from the recorded state - bypassing ``__init__``, as unpickling
does - calls the method, and compares the return value and the state after. To leave
attributes out of the record (a connection, a cache), name them in ``exclude_attrs``, or
name the ones to keep in ``state_attrs``; either needs ``factory=``, a zero-argument
callable that builds an instance for replay to set the recorded attributes on. See
:doc:`replay`.

Replay the records
------------------

Replay reruns every recorded call against the current code: it imports the function by
its recorded module and name, calls it on the recorded arguments, and compares the
result with the record. In a pytest module,
:func:`~regression_records.replay_tests` gives one test per record:

.. code-block:: python

   # tests/test_regressions.py
   from regression_records import replay_tests

   test_regressions = replay_tests("tests/regression_data")

A failing test names the function and the record, and lists every value that moved, by
path. Run it without the switch, locally and in CI; replay never writes, and nothing
records while it runs.

For a report rather than a test suite, call :func:`~regression_records.replay`:

.. code-block:: python

   from regression_records import replay

   report = replay("tests/regression_data")
   assert report.passed, report.summary()

:doc:`replay` describes the report and what replay cannot see.

Re-record after an intended change
----------------------------------

Records are never replaced. When behaviour changed on purpose, re-record with the switch
on: the new behaviour is written as a new record beside the old one, and the old one now
fails replay. Delete the old record file - the failure names its path - and then clear
the data files it leaves behind with
``RecordStore("tests/regression_data").prune(dry_run=False)``. The store's git diff shows
exactly what you accepted.

Tolerances
----------

Floats rarely reproduce bit for bit across machines and library versions.
:class:`~regression_records.ToleranceConfig` sets ``rtol`` / ``atol`` globally and per
path, and is passed to replay, not to the decorator:

.. code-block:: python

   from regression_records import ToleranceConfig, replay_tests

   tolerance = ToleranceConfig(
       rtol=1e-9,
       path_overrides={"$.output['price']": ToleranceConfig(rtol=1e-4)},
   )
   test_regressions = replay_tests("tests/regression_data", tolerance=tolerance)

Tolerance applies to *results*, never to arguments: calls with slightly different
arguments are different records.

Where next
----------

* :doc:`auto_examples/index` - fifteen runnable pages in five sections, from a first
  record through custom serializers to a realistic pipeline, ``replay_tests`` and the
  gotchas.
* :doc:`replay` - what replay does, its report, and what it cannot see.
* :doc:`comparison` - what is compared how, diffs, paths and tolerances.
* :doc:`persistence` - the store layout, identity, file naming, portability and trust.
* :doc:`limitations` - what it does not do, and what to watch for.
* :doc:`api` - the full signatures.
