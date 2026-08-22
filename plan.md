# regression-records — working notes

The package is complete against its original design (local copy:
`~/.claude/plans/objective-take-the-cozy-puppy.md`). **351 tests passing.**

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
