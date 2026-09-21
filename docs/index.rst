regression-records
==================

Record what a Python function or method did, and check on every later run that it still
does. Decorate a callable, give it a directory, and each distinct set of arguments
becomes a **regression record**: the normalized arguments, the result, and - for a
method - the object's state before and after the call. Run it again and the fresh result
is compared against the record, difference by difference, within tolerances you choose.

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
