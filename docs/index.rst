regression-records
==================

Record what a Python function or method did, and check later - after a code change, a
library upgrade, on another machine - that it still does. Decorate a callable, give it
a directory, and switch recording on with ``REGRESSION_RECORDS=1``: each distinct call
becomes a **regression record** of its normalized arguments, its result, and - for a
method - the object's state before and after. ``replay()`` then reruns every record
against the current code and compares the fresh result with the recorded one,
difference by difference, within tolerances you choose.

**Records are JSON you can read and review**, with arrays and dataframes in
content-addressed ``.npy`` / ``.parquet`` files beside them. **Comparison collects every
difference** rather than stopping at the first. **numpy is the only hard dependency**;
polars and pandas support switch on when the library is installed.

The examples are executed at build time, so every record and diff on those pages was
produced by the code, not typed in.

.. toctree::
   :maxdepth: 2

   getting_started
   auto_examples/index
   replay
   comparison
   persistence
   limitations
   design_notes
   api
   building

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
