.. _examples:

Examples
========

Fifteen pages, in reading order, each runnable on its own. Every one is executed when
these docs are built, so every record, report and figure on them was produced by the
code, not typed in.

Each page writes its code under test to a module in a fresh temporary directory, with
its store beside it, and prints where that is. Replay imports a recorded function by
module and name, and code defined on the page itself would belong to ``__main__``,
which cannot be imported again; in your project, that module is simply your package.
So a page needs nothing but ``regression-records``, numpy and matplotlib (the dataframe
page also needs polars and pandas, the ``replay_tests`` page pytest), and its ``.py``
and notebook downloads, at the foot of every page, run as they are.
