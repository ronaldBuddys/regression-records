# regression-records — working notes

The package is complete against its original design (local copy:
`~/.claude/plans/objective-take-the-cozy-puppy.md`). **379 tests passing.**

This file is no longer the design record — [README.md](README.md),
[docs/persistence.md](docs/persistence.md) and [docs/comparison.md](docs/comparison.md)
are, and every decision taken during implementation is written up in one of them. What
stays here is the residue that has nowhere else to live: options that were considered and
rejected, things examined and deliberately left alone, and work not yet done.

## What was built

| | | |
|---|---|---|
| M1 | Serialization core | `serialization/` — protocol, registry, tree walker, content-addressed naming; numpy, polars, pandas, stdlib and generic-object serializers |
| M2 | Record model + persistence | `records.py`, `store.py`, `identity.py`, `exceptions.py` |
| M3 | Comparison + tolerance | `comparison/` — `compare()`, `ComparisonResult`, `Diff`, `ToleranceConfig`, per-type comparators |
| M4 | Matching | `matching.py` — `input_hash`, lookup, `on_new_case` policies; deferred side-file writes |
| M5 | `record_func` | `_engine.py`, `decorators.py`, `CodeDriftWarning` |
| M6 | `record_method` | `_introspect.py` — receiver split, `StateSpec`, deep-copied state capture |
| M7 | Public API, packaging, docs | `__all__` (30 names), `py.typed`, classifiers, `README.md` + `docs/*.md` |
| M8 | Examples | `examples/` - eleven jupytext notebooks, run end to end by `tests/test_examples.py` |

Two `/simplify` cleanup passes ran, after M5 and during M7. The second one resolved all
six findings the first had deferred; the four that were applied are visible in the code
(`register(..., fallback=True)` on both registries, tuple/set/bytes as real serializers,
`NumpyScalarComparator`, and a collapsed `decide_action`). The two that were rejected are
below.

## Decisions with no home in the docs

**Rejected `/simplify` findings.** Both have the reasoning as a code comment at the site,
so they read as deliberate rather than overlooked — recorded here so they are not
re-raised as fresh findings:

- *Exact-type primitive tests in `to_json_tree`* (would delete the `np.generic` special
  case). Rejected: it would also reroute every other builtin subclass — `IntEnum`,
  `StrEnum`, anything subclassing `int` for units — to the generic object serializer,
  where an Enum member fails to record at all.
- *`_EXCLUDED` in `comparison/objects.py` restates registry precedence.* Rejected: the
  premise is false. `_Walker.walk` consults the registry *before* its structural
  branches, deliberately, so a user-registered comparator for a dict or list subclass
  wins. That is what makes the exclusions load-bearing, and flipping the dispatch order
  would break that extension point to delete six list entries.

**Not exported, on purpose.** `compare_exceptions` takes an exception *type name* as a
string — the shape the engine wants, a strange one to hand a user. The record model
(`Invocation`, `SerializedValue`, `ExceptionRecord`, `SchemaInfo`) and
`comparison.paths` are importable but unexported: documented as stable enough to inspect
a loaded record with, not as `__all__`.

**A correction to the original plan.** It called for polars/pandas values to raise a
`SerializationError` naming the missing extra on a no-extras install. That case is
unreachable — without polars you cannot construct a `pl.DataFrame` to pass. The real
guarantee, and the one the smoke test checks, is that the bare install imports and
records numpy end to end with the optional serializers simply absent.

**Examined during the efficiency pass and left alone**, so the next pass need not redo
the analysis:

- Re-reading a record's JSON and side files on every validated call. A cache would need
  invalidation, and consecutive calls generally hit *different* records anyway.
- The double directory glob in `find_match` — only on the new-case path, which then
  writes a file regardless.
- `registry.find()` per node in the comparison walker. ~5 `isinstance` pairs, immaterial
  for record-sized trees, and short-circuiting it would break the extension point above.

## Decisions taken while writing the examples

**Format: jupytext percent, not `.ipynb`.** The files are plain Python that runs as a
script, with `# %%` cells and `# %% [markdown]` prose. They open as notebooks in Jupyter
and VS Code and render as notebooks in Sphinx via MyST-NB's `nb_custom_formats`, but they
diff as source and can be executed in the test suite. Checked-in `.ipynb` would give
none of that.

**`tests/test_examples.py` runs each one in a subprocess.** Documentation that executes
is the only kind that cannot go stale silently. A subprocess per example, rather than
importing them, because each defines module-level decorated functions and one registers
serializers - shared state that would leak between them in-process.

**Sample classes live in `examples/_shared/models.py`, not in the example files.** A
class recorded by attribute state is re-imported by name on load, so a class defined in
`__main__` - which a script or a notebook cell is - cannot be recorded at all. The
examples would otherwise be demonstrating the one thing that does not work.

Two gaps the examples surfaced, both fixed rather than documented around:

- **`SerializerRegistry.copy()` / `ComparatorRegistry.copy()`.** The README advised
  preferring `serializer_registry=` over global `register_serializer`, but there was no
  way to obtain a registry that was the default plus one entry - only an empty one. The
  advice was unfollowable. `copy()` shares the serializer instances (they are stateless
  dispatchers) and duplicates only the ordering.
- **`path_overrides` naming a dataframe column had no effect**, though
  `docs/comparison.md` gave `"$.output['estimate']"` as its worked example. A dataframe
  comparator walks its own columns instead of recursing through `_Walker.walk`, so it
  received one tolerance resolved at the frame's path and applied it to every column. The
  resolution now happens in `compare_columns`, which is the one place every dataframe
  comparator's columns pass through. Residual limitation, not worth the protocol change
  it would cost: an override on the frame's *own* path still shadows a deeper override on
  one of its columns, because tolerance is resolved once per node rather than threaded
  from the root.

## Open items

**Unbounded diff retention.** `wrapper.last_result` holds the last `ComparisonResult` for
the process lifetime. `Diff`s hold capped reprs, and array/dataframe diffs are capped in
count — but dict comparison is not, so two fully-differing 100k-key dicts retain ~100k
diffs (~20 MB) on the wrapper indefinitely. A cap contradicts the documented "collects
every difference" contract, so this wants a deliberate decision: a global cap, an
explicit `max_diffs` on `compare()`, or leaving it and documenting the shape.

## Candidate next work

Nothing is blocking. In rough order of value:

1. **A pytest plugin or fixture** so `mode` and `overwrite` come from command-line flags
   rather than from edited decorator arguments. This is how the accept-a-change workflow
   actually wants to be driven — `pytest --regression-accept` beats editing a decorator,
   running once, and remembering to edit it back.
2. **Resolve the diff-retention item above**, most likely as `max_diffs` on `compare()`.
3. **A prune helper** for `data/` files no longer referenced by any record. They
   accumulate whenever a record is overwritten, and nothing collects them.
4. **Read `schema_version` on load**, so a future format change has somewhere to hook. It
   is stamped on every record today and never inspected.
