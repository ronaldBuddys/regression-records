# regression-records implementation plan

Full design: see conversation / `/home/buddy/.claude/plans/objective-take-the-cozy-puppy.md`.

## Open items for review

### `RunMode.VALIDATE_ONLY` + `OnNewCase.RECORD` is contradictory

The mode forbids writing; the policy asks to write. M4 resolved this by making **mode
authoritative** — an unmatched input under `VALIDATE_ONLY` raises
`UnmatchedInvocationError` naming both settings. The weakness: `OnNewCase.RECORD` is the
*default*, so a user who only switched to `VALIDATE_ONLY` gets an error citing a setting
they never chose. The error can't tell "didn't say" apart from "asked for something
impossible".

**Recommendation: default `on_new_case=None`, meaning mode-appropriate** — resolving to
`RECORD` under `RECORD_ONLY`/`BOTH` and `ERROR` under `VALIDATE_ONLY`. An *explicit*
`VALIDATE_ONLY` + `OnNewCase.RECORD` then becomes an unambiguous contradiction, and
should fail at **decoration time** with `ValueError`, not on some later call that happens
to hit novel inputs. Config errors caught when the decorator is applied beat config
errors caught in CI three weeks later.

This is the only combination where the two settings genuinely conflict, and `None` is
what encodes the distinction the current design is missing. Cost is one nullable default
plus a `resolve_on_new_case(mode, on_new_case)` helper; the resolution point is kept in
one place in `decorators.py` so adopting this stays a small change.

## Milestones

- [x] M1 — Serialization core (primitives, numpy, polars, pandas, generic object; registry; naming) — 58 tests passing
- [x] M2 — Record model + persistence (`RegressionRecord`, `RecordStore`) — 68 tests passing total
- [x] M3 — Comparator + tolerance (`compare()`, `ComparisonResult`, `Diff`, `ToleranceConfig`) — 141 tests passing total
- [x] M4 — Matching (`input_hash`, lookup, `on_new_case` policies) — 176 tests passing total
      Also landed the two issues raised in the M3 review:
      1. `SerializationContext(deferred=True)` buffers side-file writes until `flush()`
         (or `discard()`), so hashing an invocation's arguments no longer leaves
         orphaned .npy/.parquet files behind when no record ends up being written.
      2. Record filenames are now `rec-<input_hash[:16]>-<record_id[:8]>.json`, making
         `find_by_input_hash` a glob instead of a full scan; added `has_records()`.
      `config.py` (RunMode/OnNewCase/OverwritePolicy) landed here rather than M5, since
      matching needs the policy enums.
- [x] M5 — `record_func` decorator — 220 tests passing total
      Landed `_engine.py` (the shared record/validate pipeline, with a `capture_state`
      hook M6 will use), `decorators.py`, and `CodeDriftWarning`. Decisions taken here,
      all tested, all needing a mention in the M7 docs:
      - Arguments are normalized via `signature.bind()` + `apply_defaults()`, so
        `f(2, 3)`, `f(2, b=3)` and `f(a=2, b=3)` are one recorded case.
      - Unserializable args or output raise `SerializationError` rather than silently
        skipping the recording — a regression suite that quietly records nothing passes
        vacuously. `enabled=False` is the escape hatch.
      - Accepting a change via `OverwritePolicy.IF_MISMATCH` updates the record in place
        (same `record_id`) and does *not* raise: rewriting the baseline is the act of
        accepting. The diffs stay on `wrapper.last_result`.
      - The wrapped callable's own exception always wins over
        `RegressionMismatchError`, so an `exception_mismatch` is reported via
        `last_result` rather than replacing what the caller sees. A regression suite
        still fails, because the real exception propagates.
      - `RECORD_ONLY` + existing match + `OverwritePolicy.NEVER` writes nothing, rather
        than duplicating the case under a second record id.
      - `KeyboardInterrupt`/`SystemExit` propagate unrecorded; only `Exception` is
        captured.
- [x] Cleanup pass (`/simplify`) after M5 — still 220 tests passing, no behavior change
      intended. Reuse/duplication: `describe_positions`, `require_both`, `compare_columns`
      and `safely_equal` moved into `comparison/base.py`, replacing three copies of the
      capped-index renderer, four copies of the "only one side is a DataFrame" guard, two
      copies of the per-column truncation loop, and two copies of the guarded-`==`
      fallback. `records.to_dict/from_dict` now use `dataclasses.asdict`, and `identity.py`
      calls `naming.hash_bytes` instead of inlining `sha256(...)[:16]` twice.
      Layering: `comparison/paths.py` now owns the diff-path grammar (`ROOT_PATH`,
      `child`/`index`/`column`/`is_under`, and the three record roots), which
      `ToleranceConfig.for_path` and the engine both consume instead of re-deriving;
      `comparison/exception_diffs.py` owns exception-vs-success wording, so the engine no
      longer authors `Diff`s by hand; `OnMismatch` moved to `config.py` beside its three
      sibling policy enums, validated in `RecorderConfig.__post_init__` rather than at the
      decorator call site.
      One real inconsistency surfaced and was fixed: the serializer and the comparator
      each had their own copy of "what counts as an object's state", and they disagreed.
      `_object_state.py` is now the single definition. Its `__getstate__`-answers-`None`
      rule is what keeps functions and lambdas unrecordable - they have a `__dict__`, but
      an empty one, so a `vars()` fallback would have recorded them as empty-state objects
      instead of raising `SerializationError`.
      Deliberately not done, with reasons: `matching.decide_action` still raises rather
      than returning unmatched-ness as data (it interacts with the open review item above,
      so restructuring it now would churn code that item may change anyway); numpy is
      still special-cased inside both walkers rather than registry-driven (the fix,
      exact-type primitive tests, reroutes `IntEnum` and other builtin subclasses to the
      object serializer - a behavior change that needs its own commit and tests);
      `_EXCLUDED` in `comparison/objects.py` still restates comparator precedence; and the
      registries' "insert just before the fallback" rule is still caller-side arithmetic
      in two places rather than a `register(..., fallback=True)` concept.
- [x] M6 — `record_method` decorator — 242 tests passing total
      Landed `_introspect.py` (`split_receiver`, `StateSpec`, `capture_state`) and
      `record_method`, reusing `_build_decorator` and the `capture_state` hook M5 left in
      `_engine.execute`. Decisions taken here, all tested, all needing M7 docs:
      - `self` is excluded from the recorded arguments *and* from the input hash. The
        object is described by its captured state instead; hashing the instance would
        make every instance - and, since most objects hash by identity, every run - a
        separate case. Consequence worth documenting: two instances in different states
        calling with the same arguments are *one* case, and the state difference is
        reported as a mismatch rather than recorded as a second case.
      - Remaining arguments are recorded by name (`args` is always `[]`). With the
        receiver removed, positional order no longer describes the call, and
        `apply_defaults()` has already named everything.
      - `state_before` is deep-copied *before* the call - the reason this decorator
        exists. `state_after` is copied too, for one code path; the copy costs far less
        than the serialization that follows it.
      - A method that mutates its object is regression-tested on the mutation even when
        its return value is unchanged. That is the failure `record_func` structurally
        cannot see.
      - State is captured after a call that raised, too: a method that failed part-way
        may still have mutated its object, and that mutation is the interesting part.
      - An uncopyable attribute (a connection, a file handle) raises `SerializationError`
        naming `exclude_attrs` as the escape hatch, rather than silently dropping it.
      - A receiver with no dict-shaped state (a `__slots__`-only class) is rejected
        clearly; a receiver whose `__dict__` is merely *empty* captures `{}`, since
        "nothing assigned yet" is a legitimate starting state.
- [ ] M7 — Public API + packaging + docs + full-suite verification

  ### Public API (`src/regression_records/__init__.py`)
  Currently exports only the exception types and `__version__`; `record_func` and
  `record_method` are reachable only via `regression_records.decorators`, which no user
  should have to know. Export, with `__all__`:
  - decorators: `record_func`, `record_method`
  - config enums: `RunMode`, `OnNewCase`, `OnMismatch`, `OverwritePolicy`
  - store/model: `RecordStore`, `RegressionRecord`
  - comparison: `compare`, `ComparisonResult`, `Diff`, `ToleranceConfig`
  - extension points: `register_serializer`, `register_comparator`, and the `Serializer`
    / `Comparator` protocols - a user writing one needs the type to annotate against
  Decide explicitly whether `compare_exceptions`, `SerializerRegistry`,
  `ComparatorRegistry` and `FunctionIdentity` are public; they are all currently
  importable but undocumented, and shipping them by accident makes them API.

  ### Packaging
  - `py.typed` marker file (the package is fully annotated but currently advertises
    nothing to type checkers).
  - `pyproject.toml`: classifiers, `requires-python`, project URLs, license, and a check
    that the `polars` / `pandas` / `test` extras resolve.
  - Smoke test: `python -m build`, then in a fresh venv `pip install .` (no extras) and
    `pip install .[polars,pandas]`. The no-extras install must import cleanly and support
    numpy-only recording, with polars/pandas values raising a `SerializationError` that
    names the missing extra - *not* an `ImportError` at import time. This is the one
    verification the current test suite cannot perform, since its venv has everything.

  ### Tests still owed
  - `tests/test_exceptions.py` - each exception type in its own scenario, with message
    content checks. `RecordNotFoundError`, `UnmatchedInvocationError`, `SerializationError`
    and `RegressionMismatchError` are covered incidentally by the decorator tests;
    `DeserializationError`, `CorruptRecordError` and `IdentityResolutionError` are not
    covered from the decorator path at all.
  - Overwrite-policy edge cases: `ALWAYS` combined with `VALIDATE_ONLY` (writes on every
    call in a mode that claims not to write - decide and test which wins), and `ALWAYS`
    on a *new* case under `BOTH`.
  - `RecordStore.rename_identity` is public, documented, and has no test or caller.

  ### Documentation
  `README.md` is currently one line. Needed: concept overview, install incl. extras,
  quickstarts for both decorators, config table, the "accepting an intentional behavior
  change" worked example (`OverwritePolicy.IF_MISMATCH`), the "adding a serializer for
  your own type" worked example, and an exception reference table.
  `docs/persistence.md`: annotated directory tree, content-addressing and the
  `rec-<input_hash>-<record_id>.json` naming, the portability guarantee (relative
  refs; a store directory can be moved or copied as a unit), and the deferred-write
  mechanism that keeps a validate-only run from leaving orphaned side files.
  `docs/comparison.md`: `ComparisonResult`/`Diff` field reference, the path grammar from
  `comparison/paths.py` (it is public API - `path_overrides` keys are matched against
  it), `ToleranceConfig` incl. `path_overrides` precedence, why a dtype change is a
  failure, and the diff-capping rules.

  **Decisions accumulated in M5/M6/cleanup that the docs must carry** - each is listed
  under its milestone above: argument normalization; unserializable inputs raising rather
  than skipping; `IF_MISMATCH` not raising; the real exception always winning over
  `RegressionMismatchError`; `KeyboardInterrupt`/`SystemExit` propagating unrecorded;
  `self` excluded from the input hash and its surprising consequence (two instances in
  different states are *one* case); `state_before` deep-copied before the call;
  `exclude_attrs` as the escape hatch for uncopyable state; and `CodeDriftWarning`'s
  meaning.

  **Known limitations to document rather than silently omit**: polars `LazyFrame`
  requires `.collect()`; `dtype=object` arrays rejected; non-string dict keys rejected;
  classes defined in `__main__` rejected; the pandas parquet→pickle fallback; pandas
  `DatetimeIndex.freq` not preserved through parquet; module moves not auto-resolved;
  no concurrent/multi-process write safety; no CLI for browsing or pruning a store.

  ### Deferred from the `/simplify` pass, each needing its own change
  - `matching.decide_action` raising rather than returning unmatched-ness as data
    (blocked on the open review item at the top of this file).
  - numpy special-cased inside both walkers instead of registry-driven; the fix is exact-
    type primitive tests plus a `NumpyScalarComparator`, which reroutes `IntEnum` and
    other builtin subclasses to the object serializer - a behavior change needing tests.
  - `tuple`/`set`/`bytes` type tags dispatched by an if-chain in `from_json_tree` while
    every other tag goes through `registry.by_tag`, so a third-party serializer claiming
    one of those tags is silently shadowed.
  - `_EXCLUDED` in `comparison/objects.py` restating precedence that registry order and
    walker dispatch order already encode.
  - Both registries' "insert just before the fallback" rule is caller-side arithmetic
    (`max(len(registry) - 1, 0)`) duplicated in two packages; it should be a
    `register(..., fallback=True)` concept owned by the registries.
  - The efficiency review never ran (session limit), so that angle is uncovered: no one
    has looked for redundant per-call work, repeated store I/O, or closures pinning large
    scopes.
