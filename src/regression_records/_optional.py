"""Telling "this optional library is not installed" from "it is installed but broken"."""

from __future__ import annotations


def library_is_absent(error: ImportError, library: str) -> bool:
    """Whether ``error`` means ``library`` itself is not installed.

    Support for polars and pandas registers only when the library imports. Skipping it
    on *any* ``ImportError`` would also hide a broken install - and then a DataFrame is
    not refused: it falls through to the object path and is recorded by its internal
    block-manager attributes. So only a clean "no such module" is treated as
    absence; every other import failure propagates.
    """
    return isinstance(error, ModuleNotFoundError) and error.name == library
