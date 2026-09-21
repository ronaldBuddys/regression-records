.. _examples:

Examples
========

A worked tour, in order: recording a first call and replaying it, catching a regression
with ``replay``, tolerances, modes, accepting a change, methods, exceptions and code
drift, your own classes and serializers, dataframes and the store - then a realistic
pipeline taken through record, change and replay, the gotchas, and ``replay_tests``,
which turns a store into one pytest test per record.

Each script is standalone, runs in a few seconds under plain ``python``, and writes to a
throwaway store under the system temp directory. Each also opens as a notebook; the
download links are at the foot of every page. Examples 1 to 8 need only numpy;
example 14 also needs pytest.

The scripts share helper modules in ``examples/_shared``: ``demo.py`` (a fresh store
and the printing helpers), ``models.py`` (the sample classes, which have to live in an
importable module to be recorded - example 8 explains why), and one module per example
holding the functions it records and replays. Replay imports a recorded function by its
module and name, and a function defined in the script itself belongs to ``__main__``,
which cannot be imported again.
