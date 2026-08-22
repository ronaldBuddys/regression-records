"""Support code for the examples: sample classes, and small display helpers.

Kept in a real importable module on purpose. A class recorded by its attribute state is
stored as ``{"module": ..., "qualname": ...}`` and re-imported by name when the record is
loaded, so a class defined in ``__main__`` - which is where a script or a notebook cell
puts it - cannot be recorded at all. See ``08_custom_classes.py``.
"""
