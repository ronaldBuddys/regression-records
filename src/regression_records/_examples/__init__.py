"""Support code for the documentation and the example gallery only.

Every submodule here backs one page of the gallery under ``examples/`` or one
illustrative snippet in the hand-written docs: a function or a class that needs a real,
importable module to live in, since replay refuses anything defined in ``__main__``.

A submodule that records reads its store from the ``REGRESSION_RECORDS_STORE``
environment variable when it is imported, as a project's own code would: where records
go is configuration, set by whoever runs the code (a ``.env`` file, a CI job), not
written into it. Importing one with the variable unset raises ``KeyError``. The
variable is a convention of these examples; ``regression_records`` itself never reads
it.

**Nothing here is public API.** Signatures, names and behaviour may change, move or
disappear between releases without notice, and without a ``CHANGELOG`` entry. Do not
import from ``regression_records._examples`` in your own code; it exists so the
examples can import real functions and classes instead of writing a module to a
temporary file at each page's top.
"""
