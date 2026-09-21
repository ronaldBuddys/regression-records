.. _examples:

Examples
========

A worked tour, in order: a first recording, reading a failure, tolerances, modes,
accepting a change, methods, exceptions, your own classes and serializers, dataframes and
the store - then a realistic pipeline, and the gotchas.

Each script is standalone, runs in a few seconds under plain ``python``, and writes to a
throwaway store under the system temp directory. Each also opens as a notebook; the
download links are at the foot of every page. Examples 1 to 8 need only numpy.

The scripts share two helper modules in ``examples/_shared``: ``demo.py`` (a fresh store
and the printing helpers) and ``models.py`` (the sample classes, which have to live in
an importable module to be recorded - example 8 explains why).
