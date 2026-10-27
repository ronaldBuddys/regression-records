regression-records
==================

**Record real calls. Replay them as regression tests.**

Decorate a function or method and run your code once with recording switched on: each
call is written to a store as a reviewable JSON record, with arrays and dataframes in
content-addressed side files beside it. Later - after a refactor, a dependency upgrade,
on another machine - ``replay`` re-runs every recorded call against the current code
and reports each difference down to its exact path, within the tolerances you choose.

.. image:: _static/how-it-works.svg
   :alt: Record: a decorated function writes each call to a store of JSON records.
         Replay re-runs every recorded call against today's code and reports each
         difference by path.
   :width: 100%

In ten lines
------------

.. code-block:: python

   import numpy as np
   from regression_records import record_func, replay

   @record_func("tests/regression_data")          # 1. decorate, in your package
   def discount_curve(tenors: np.ndarray) -> np.ndarray:
       return np.exp(-0.03 * tenors)

   # 2. record:  REGRESSION_RECORDS=1 pytest   (or any run that calls it)
   # 3. change the code, then replay every recorded call against it:
   report = replay("tests/regression_data")
   assert report.passed, report.summary()

When something moved, the report says exactly what, where, and by how much - even when
the change is far too small to see:

.. image:: _static/readme/caught-regression.png
   :alt: A recorded discount curve and the curve after a change overlap exactly; the
         relative difference beyond 20 years leaves the 0.1% tolerance band, and the
         eleven breaching points are those the replay report names.
   :width: 100%

Where to go next
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install from git, record a function and a method, replay from pytest, and accept
      a change you meant to make.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: ref

      Fifteen executed pages in five sections, from a first record to dataframes, a
      pricing pipeline and the gotchas - every figure produced by the code.

   .. grid-item-card:: Concepts
      :link: replay
      :link-type: doc

      :doc:`replay` · :doc:`persistence` · :doc:`comparison` · :doc:`limitations` -
      what replay does, what the store holds, how values are compared, and what none of
      it can see.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Every public name, with its full signature and docstring.

**Records are JSON you can read and review.** **Comparison collects every difference**
rather than stopping at the first. **numpy is the only hard dependency**; polars and
pandas support switches on when the library is installed. Loading a store imports the
code it names, so treat a store as you treat code: see
`SECURITY.md <https://github.com/ronaldBuddys/regression-records/blob/main/SECURITY.md>`_
and :doc:`persistence`.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   auto_examples/index
   replay
   comparison
   persistence
   limitations
   design_notes
   api
   changelog
   building
