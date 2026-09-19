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

Decorate it and say where its history lives. There is no default location, because
recording somewhere implicit would be worse than being asked.

.. code-block:: python

   import numpy as np
   from regression_records import record_func

   @record_func("tests/regression_data")
   def discount_factors(rate: float, years: np.ndarray) -> np.ndarray:
       return np.exp(-rate * years)

   discount_factors(0.03, np.array([1.0, 2.0, 5.0]))   # first run: recorded
   discount_factors(0.03, np.array([1.0, 2.0, 5.0]))   # later runs: compared

The first call with a given set of arguments writes a record. Every later call with the
same arguments is compared against it, and a difference raises
:class:`~regression_records.RegressionMismatchError` listing every value that moved, by
path. New arguments are a new case, recorded alongside.

The function itself is untouched: it receives its original arguments, its return value
passes through, and an exception it raises is always re-raised (and recorded, so that
"raises ``ValueError``" is regression-tested too).

Record a method
---------------

A method's behaviour is as much about the object as about its arguments, so
:func:`~regression_records.record_method` also snapshots ``self`` before and after the
call. A method that returns the right answer by way of the wrong mutation is caught.

.. code-block:: python

   from regression_records import record_method

   class Portfolio:
       def __init__(self):
           self.positions = {}

       @record_method("tests/regression_data", exclude_attrs=["_cache"])
       def trade(self, symbol: str, quantity: int) -> int:
           self.positions[symbol] = self.positions.get(symbol, 0) + quantity
           return self.positions[symbol]

Use it from pytest, and validate only in CI
-------------------------------------------

Call the decorated function from an ordinary test. Locally, the default
``RunMode.BOTH`` records new cases as you write them. Commit the store directory, and
have CI refuse to record, so a case with no history fails instead of quietly becoming
one:

.. code-block:: python

   import os
   from regression_records import RunMode, record_func

   MODE = RunMode.VALIDATE_ONLY if os.environ.get("CI") else RunMode.BOTH

   @record_func("tests/regression_data", mode=MODE)
   def discount_factors(rate, years): ...

Options accept the enum member or its string value (``mode="validate_only"``); a
misspelt one raises when the decorator is applied.

Accept an intentional change
----------------------------

When behaviour changed on purpose, re-run once with
``overwrite=OverwritePolicy.IF_MISMATCH``: mismatching records are replaced, the diff
stays on ``func.last_result`` for you to read, and the store's git diff shows exactly
what you accepted. Then take the flag back off.

Tolerances
----------

Floats rarely reproduce bit for bit across machines and library versions.
:class:`~regression_records.ToleranceConfig` sets ``rtol`` / ``atol`` globally and per
path:

.. code-block:: python

   from regression_records import ToleranceConfig

   tolerance = ToleranceConfig(
       rtol=1e-9,
       path_overrides={"$.output['price']": ToleranceConfig(rtol=1e-4)},
   )

Tolerance applies to *results*, never to matching: which record a call belongs to is
decided by an exact hash of its arguments.

Where next
----------

* :doc:`auto_examples/index` - thirteen runnable scripts, from a first record through
  custom serializers to a realistic pipeline, ending with the gotchas.
* :doc:`comparison` - what is compared how, diffs, paths and tolerances.
* :doc:`persistence` - the store layout, identity, matching, portability and trust.
* :doc:`limitations` - what it does not do, and what to watch for.
* :doc:`api` - the full signatures.
