# Examples

A worked tour of regression-records, from a first recording to writing your own
serializers. Each file is standalone, runs in a few seconds, and writes to a throwaway
store under your system temp directory - nothing here touches the repository.

```bash
python examples/01_first_record.py
```

## The tour

| | | |
|---|---|---|
| 1 | [First record](01_first_record.py) | Decorate, call, and read what landed on disk. Argument binding, the input hash, validating on the second call. |
| 2 | [Catching a regression](02_catching_a_regression.py) | Break the code behind a recorded function and read the failure. Diff kinds, the path grammar, `on_mismatch`. |
| 3 | [Tolerance](03_tolerance.py) | `rtol` / `atol` / `equal_nan` / `strict_dtype`, and per-path overrides. |
| 4 | [Modes and unseen inputs](04_modes_and_new_cases.py) | `RunMode`, `OnNewCase`, and choosing a mode from the environment for CI. |
| 5 | [Accepting a change](05_accepting_a_change.py) | `OverwritePolicy`, the accept-a-change workflow, and what overwriting leaves behind. |
| 6 | [Recording methods](06_recording_methods.py) | `record_method`, state before and after, `state_attrs` / `exclude_attrs`. |
| 7 | [Exceptions and drift](07_exceptions_and_drift.py) | Recording what a function raises, and `CodeDriftWarning`. |
| 8 | [Custom classes](08_custom_classes.py) | `to_json_tree` / `from_json_tree` directly, `__dict__` and `__getstate__` classes, nesting. |
| 9 | [Custom serializer](09_custom_serializer.py) | Writing a `Serializer` and a `Comparator`, inline and side-file storage, scoped registries. |
| 10 | [DataFrames](10_dataframes.py) | polars and pandas, schema-first comparison, column tolerances, the pickle fallback. |
| 11 | [The store](11_inspecting_the_store.py) | `RecordStore`, reading recorded values back, tags, `rename_identity`, using it from pytest. |

Examples 1-8 need only numpy. Example 10 skips whichever of polars/pandas is missing.

## Format

These are [jupytext](https://jupytext.readthedocs.io) *percent-format* notebooks: plain
Python that runs as a script, with `# %%` cell markers and `# %% [markdown]` prose cells.
The same file opens as a notebook in Jupyter and VS Code, and renders as one in Sphinx.

To convert:

```bash
pip install jupytext
jupytext --to ipynb examples/*.py
```

To render them in a Sphinx site with [MyST-NB](https://myst-nb.readthedocs.io), no
conversion is needed - point it at the `.py` files:

```python
# docs/conf.py
extensions = ["myst_nb"]
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_mode = "auto"
```

Run notebooks with `examples/` as the working directory, so the `_shared` helpers import.

## `_shared/`

Two support modules, not part of the tour:

- **`_shared/models.py`** - the sample classes. They live in an importable module on
  purpose: a class recorded by its attribute state is re-imported by name when the record
  is loaded, so a class defined in `__main__` (a script, or a notebook cell) cannot be
  recorded at all. Example 8 covers this.
- **`_shared/demo.py`** - `fresh_store()` and the printing helpers, so the examples spend
  their lines on the library rather than on formatting output.
