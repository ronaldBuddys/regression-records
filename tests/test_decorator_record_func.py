import numpy as np
import polars as pl
import pytest

from regression_records.comparison import ToleranceConfig
from regression_records.config import OnNewCase, OverwritePolicy, RunMode
from regression_records.decorators import record_func
from regression_records.exceptions import (
    CodeDriftWarning,
    RecordNotFoundError,
    RegressionMismatchError,
    SerializationError,
    UnmatchedInvocationError,
)
from regression_records.store import RecordStore


def add(a, b=1, *, scale=1):
    """Add two numbers, then scale."""
    return (a + b) * scale


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path)


def records_for(store, wrapper):
    return store.list_records(wrapper.identity)


# ------------------------------------------------------------------ transparency


def test_return_value_passes_through(store):
    recorded = record_func(store)(add)
    assert recorded(2, 3) == 5
    assert recorded(2, 3, scale=2) == 10


def test_functools_wraps_metadata_preserved(store):
    recorded = record_func(store)(add)
    assert recorded.__name__ == "add"
    assert recorded.__doc__ == "Add two numbers, then scale."
    assert recorded.__wrapped__ is add


def test_exception_is_always_reraised(store):
    def boom(x):
        raise ValueError(f"bad {x}")

    recorded = record_func(store)(boom)
    with pytest.raises(ValueError, match="bad 1"):
        recorded(1)


def test_wrong_arguments_still_raise_the_real_type_error(store):
    recorded = record_func(store)(add)
    with pytest.raises(TypeError):
        recorded(1, 2, 3, 4)
    assert records_for(store, recorded) == []


def test_disabled_decorator_is_a_pure_passthrough(tmp_path):
    store = RecordStore(tmp_path / "store")
    recorded = record_func(store, enabled=False)(add)
    assert recorded(2, 3) == 5
    assert not (tmp_path / "store").exists()
    assert recorded.last_result is None


def test_store_accepts_a_path(tmp_path):
    recorded = record_func(tmp_path / "records")(add)
    assert recorded(1, 1) == 2
    assert (tmp_path / "records").exists()


# --------------------------------------------------------------------- recording


def test_first_call_records_the_invocation(store):
    recorded = record_func(store)(add)
    recorded(2, 3)

    written = records_for(store, recorded)
    assert len(written) == 1
    ctx = store.deserialization_context(recorded.identity)
    assert written[0].load_output(ctx) == 5
    assert written[0].load_args(ctx) == [2, 3]
    assert written[0].load_kwargs(ctx) == {"scale": 1}


def test_equivalent_call_styles_share_one_record(store):
    recorded = record_func(store)(add)
    recorded(2, 3)
    recorded(2, b=3)
    recorded(a=2, b=3)
    recorded(2, 3, scale=1)

    assert len(records_for(store, recorded)) == 1


def test_distinct_inputs_produce_distinct_records(store):
    recorded = record_func(store)(add)
    recorded(1, 1)
    recorded(2, 2)
    recorded(1, 1, scale=3)

    assert len(records_for(store, recorded)) == 3


def test_repeated_matching_call_does_not_duplicate(store):
    recorded = record_func(store)(add)
    for _ in range(5):
        recorded(2, 3)
    assert len(records_for(store, recorded)) == 1


def test_tags_are_stored(store):
    recorded = record_func(store, tags={"suite": "smoke"})(add)
    recorded(1, 1)
    assert records_for(store, recorded)[0].tags == {"suite": "smoke"}


def test_complex_arguments_and_return_values(store):
    def transform(frame, weights, meta):
        return {
            "frame": frame.with_columns(pl.col("a") * 2),
            "total": weights.sum(),
            "meta": meta,
        }

    recorded = record_func(store)(transform)
    args = (pl.DataFrame({"a": [1, 2]}), np.array([1.0, 2.0]), {"n": (1, 2)})
    first = recorded(*args)
    assert recorded(*args)["total"] == first["total"]
    assert len(records_for(store, recorded)) == 1


def test_none_return_value_is_recorded_and_validated(store):
    def returns_none(x):
        return None

    recorded = record_func(store)(returns_none)
    assert recorded(1) is None
    assert recorded(1) is None
    assert records_for(store, recorded)[0].succeeded


# -------------------------------------------------------------------- validation


def test_seed_then_validate_passes(store):
    recorded = record_func(store)(add)
    recorded(2, 3)
    assert recorded(2, 3) == 5
    assert recorded.last_result.passed


def test_changed_behavior_raises_a_mismatch(store):
    def compute(x):
        return x * 2

    seeded = record_func(store)(compute)
    seeded(5)

    def compute(x):  # noqa: F811 - deliberately a different implementation
        return x * 3

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(compute)
    with pytest.raises(RegressionMismatchError) as excinfo:
        validating(5)

    result = excinfo.value.result
    assert not result.passed
    assert result.diffs[0].path == "$.output"
    assert "15" in result.diffs[0].actual


def test_on_mismatch_return_does_not_raise(store):
    def compute(x):
        return x * 2

    record_func(store)(compute)(5)

    def compute(x):  # noqa: F811
        return x * 3

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY, on_mismatch="return")(
        compute
    )
    assert validating(5) == 15
    assert not validating.last_result.passed


def test_invalid_on_mismatch_is_rejected_at_decoration(store):
    with pytest.raises(ValueError, match="on_mismatch"):
        record_func(store, on_mismatch="explode")


def test_tolerance_absorbs_small_numeric_drift(store):
    def compute(x):
        return x * 3.0

    record_func(store)(compute)(1.0)

    def compute(x):  # noqa: F811
        return x * 3.0 + 1e-9

    tolerant = record_func(
        store, mode=RunMode.VALIDATE_ONLY, tolerance=ToleranceConfig(atol=1e-6)
    )(compute)
    assert tolerant(1.0) == pytest.approx(3.0)
    assert tolerant.last_result.passed

    strict = record_func(
        store, mode=RunMode.VALIDATE_ONLY, tolerance=ToleranceConfig(rtol=0.0)
    )(compute)
    with pytest.raises(RegressionMismatchError):
        strict(1.0)


def test_array_output_mismatch_reports_a_useful_diff(store):
    def make(n):
        return np.zeros(n)

    record_func(store)(make)(4)

    def make(n):  # noqa: F811
        return np.ones(n)

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY, on_mismatch="return")(
        make
    )
    validating(4)
    diff = validating.last_result.diffs[0]
    assert diff.kind == "value_mismatch"
    assert "4 of 4 element(s) differ" in diff.message


def test_validate_only_without_records_raises(store):
    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(add)
    with pytest.raises(RecordNotFoundError):
        validating(1, 1)


def test_validate_only_writes_nothing_for_a_passing_call(store):
    record_func(store)(add)(2, 3)
    before = sorted(p.name for p in store.root.rglob("*") if p.is_file())

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(add)
    validating(2, 3)

    assert sorted(p.name for p in store.root.rglob("*") if p.is_file()) == before


def test_validate_only_leaves_no_orphan_data_for_a_new_case(store):
    def identity_fn(arr):
        return arr

    record_func(store)(identity_fn)(np.arange(3))
    before = sorted(p.name for p in store.root.rglob("*") if p.is_file())

    validating = record_func(
        store, mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.IGNORE
    )(identity_fn)
    validating(np.arange(99))

    assert sorted(p.name for p in store.root.rglob("*") if p.is_file()) == before


# ---------------------------------------------------------------------- new cases


def test_on_new_case_error(store):
    recorded = record_func(store)(add)
    recorded(1, 1)

    strict = record_func(store, on_new_case=OnNewCase.ERROR)(add)
    assert strict(1, 1) == 2
    with pytest.raises(UnmatchedInvocationError):
        strict(5, 5)


def test_on_new_case_ignore_records_nothing(store):
    ignoring = record_func(store, on_new_case=OnNewCase.IGNORE)(add)
    assert ignoring(1, 1) == 2
    assert records_for(store, ignoring) == []
    assert ignoring.last_result is None


def test_unset_on_new_case_records_under_both(store):
    recorded = record_func(store)(add)
    assert recorded(1, 1) == 2
    assert len(records_for(store, recorded)) == 1


def test_unset_on_new_case_errors_under_validate_only(store):
    record_func(store)(add)(1, 1)

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(add)
    assert validating(1, 1) == 2  # the recorded case still validates
    with pytest.raises(UnmatchedInvocationError, match="VALIDATE_ONLY"):
        validating(9, 9)


def test_contradicting_the_mode_fails_when_the_decorator_is_applied(store):
    # Not on some later call that happens to reach novel inputs - by then the run is
    # minutes old and the traceback points at a call site, not at the misconfiguration.
    with pytest.raises(ValueError, match="contradicts"):
        record_func(store, mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.RECORD)


# --------------------------------------------------------------- overwrite policy


def test_overwrite_never_keeps_the_original_record(store):
    def compute(x):
        return x * 2

    record_func(store)(compute)(5)

    def compute(x):  # noqa: F811
        return x * 3

    validating = record_func(store, on_mismatch="return")(compute)
    validating(5)

    ctx = store.deserialization_context(validating.identity)
    stored = store.list_records(validating.identity)
    assert len(stored) == 1
    assert stored[0].load_output(ctx) == 10


def test_overwrite_if_mismatch_accepts_the_new_behavior(store):
    def compute(x):
        return x * 2

    seeded = record_func(store)(compute)
    seeded(5)
    original_id = store.list_records(seeded.identity)[0].record_id

    def compute(x):  # noqa: F811
        return x * 3

    accepting = record_func(store, overwrite=OverwritePolicy.IF_MISMATCH)(compute)
    assert accepting(5) == 15  # accepting the change resolves it; no error raised
    assert not accepting.last_result.passed

    stored = store.list_records(accepting.identity)
    assert len(stored) == 1
    assert stored[0].record_id == original_id  # updated in place, not duplicated
    assert (
        stored[0].load_output(store.deserialization_context(accepting.identity)) == 15
    )

    # And the updated record is now the baseline.
    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(compute)
    assert validating(5) == 15
    assert validating.last_result.passed


def test_overwrite_if_mismatch_leaves_a_passing_record_untouched(store):
    recorded = record_func(store, overwrite=OverwritePolicy.IF_MISMATCH)(add)
    recorded(2, 3)
    path = next(store._records_dir(recorded.identity).glob("*.json"))
    original = path.read_text()

    recorded(2, 3)
    assert path.read_text() == original


def test_overwrite_always_rewrites_a_matching_record(store):
    recorded = record_func(store, overwrite=OverwritePolicy.ALWAYS)(add)
    recorded(2, 3)
    first = next(store._records_dir(recorded.identity).glob("*.json")).read_text()

    recorded(2, 3)
    second = next(store._records_dir(recorded.identity).glob("*.json")).read_text()
    assert first != second  # created_at moved on
    assert len(records_for(store, recorded)) == 1


def test_overwrite_always_records_a_new_case_rather_than_rewriting_another(store):
    recorded = record_func(store, overwrite=OverwritePolicy.ALWAYS)(add)
    recorded(2, 3)
    recorded(4, 5)
    # ALWAYS governs replacing the record for *these* inputs; a case that has never
    # been seen is still a new record, not an overwrite of an unrelated one.
    assert len(records_for(store, recorded)) == 2


def test_overwrite_always_rebaselines_under_validate_only(store):
    """RunMode gates *new* cases; OverwritePolicy gates replacing existing ones.

    They are orthogonal, so ALWAYS + VALIDATE_ONLY is coherent and useful: re-baseline
    every case already on record, without inventing baselines for inputs nobody has
    recorded yet.
    """

    def compute(x):
        return x * 2

    record_func(store)(compute)(5)
    identity = record_func(store)(compute).identity

    def compute(x):  # noqa: F811
        return x * 3

    rebaselining = record_func(
        store,
        mode=RunMode.VALIDATE_ONLY,
        overwrite=OverwritePolicy.ALWAYS,
        name=identity.qualname,
    )(compute)
    assert rebaselining(5) == 15  # accepted, not raised
    assert len(records_for(store, rebaselining)) == 1

    # ... and the new baseline is what a later validation compares against.
    validating = record_func(store, mode=RunMode.VALIDATE_ONLY, name=identity.qualname)(
        compute
    )
    assert validating(5) == 15
    assert validating.last_result.passed

    # A case nobody recorded is still refused rather than silently created.
    with pytest.raises(UnmatchedInvocationError):
        rebaselining(99)


def test_record_only_does_not_duplicate_an_existing_case(store):
    recorded = record_func(store, mode=RunMode.RECORD_ONLY)(add)
    recorded(2, 3)
    recorded(2, 3)
    assert len(records_for(store, recorded)) == 1


def test_record_only_never_validates(store):
    def compute(x):
        return x * 2

    record_func(store)(compute)(5)

    def compute(x):  # noqa: F811
        return x * 3

    recording = record_func(store, mode=RunMode.RECORD_ONLY)(compute)
    assert recording(5) == 15  # no mismatch raised
    assert recording.last_result is None


# --------------------------------------------------------------------- exceptions


def test_exceptions_are_recorded_by_default(store):
    def boom(x):
        raise ValueError("nope")

    recorded = record_func(store)(boom)
    with pytest.raises(ValueError):
        recorded(1)

    stored = records_for(store, recorded)
    assert len(stored) == 1
    assert not stored[0].succeeded
    assert stored[0].exception.exc_type == "ValueError"
    assert stored[0].exception.message == "nope"
    assert "raise ValueError" in stored[0].exception.traceback_text


def test_a_consistently_raised_exception_validates_clean(store):
    def boom(x):
        raise ValueError("nope")

    recorded = record_func(store)(boom)
    for _ in range(2):
        with pytest.raises(ValueError):
            recorded(1)
    assert recorded.last_result.passed
    assert len(records_for(store, recorded)) == 1


def test_exception_message_change_is_not_a_mismatch(store):
    def boom(x):
        raise ValueError("first message")

    recorded = record_func(store)(boom)
    with pytest.raises(ValueError):
        recorded(1)

    def boom(x):  # noqa: F811
        raise ValueError("a totally different message")

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(boom)
    with pytest.raises(ValueError):
        validating(1)
    assert validating.last_result.passed


def test_exception_type_change_is_a_mismatch(store):
    def boom(x):
        raise ValueError("nope")

    recorded = record_func(store)(boom)
    with pytest.raises(ValueError):
        recorded(1)

    def boom(x):  # noqa: F811
        raise KeyError("nope")

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(boom)
    # The real exception always reaches the caller, even when it is also a regression.
    with pytest.raises(KeyError):
        validating(1)
    assert not validating.last_result.passed
    assert validating.last_result.diffs[0].kind == "exception_mismatch"


def test_newly_raised_exception_where_a_value_was_recorded(store):
    def compute(x):
        return x * 2

    record_func(store)(compute)(5)

    def compute(x):  # noqa: F811
        raise RuntimeError("regressed")

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(compute)
    with pytest.raises(RuntimeError):
        validating(5)
    diff = validating.last_result.diffs[0]
    assert diff.kind == "exception_mismatch"
    assert "returned a value" in diff.message


def test_newly_returning_where_an_exception_was_recorded(store):
    def compute(x):
        raise RuntimeError("broken")

    recorded = record_func(store)(compute)
    with pytest.raises(RuntimeError):
        recorded(5)

    def compute(x):  # noqa: F811
        return x

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY, on_mismatch="return")(
        compute
    )
    assert validating(5) == 5
    assert validating.last_result.diffs[0].kind == "exception_mismatch"


def test_record_exceptions_false_records_nothing_on_failure(store):
    def boom(x):
        raise ValueError("nope")

    recorded = record_func(store, record_exceptions=False)(boom)
    with pytest.raises(ValueError):
        recorded(1)
    assert records_for(store, recorded) == []


def test_keyboard_interrupt_propagates_without_being_recorded(store):
    def interrupted(x):
        raise KeyboardInterrupt

    recorded = record_func(store)(interrupted)
    with pytest.raises(KeyboardInterrupt):
        recorded(1)
    assert records_for(store, recorded) == []


def test_unserializable_argument_raises(store):
    recorded = record_func(store)(add)
    with pytest.raises(SerializationError):
        recorded(lambda: None, 1)


def test_unserializable_output_raises(store):
    def make_lambda(x):
        return lambda: x

    recorded = record_func(store)(make_lambda)
    with pytest.raises(SerializationError):
        recorded(1)
    assert records_for(store, recorded) == []


# ----------------------------------------------------------------------- identity


def test_same_name_in_different_modules_do_not_collide(store):
    def compute(x):
        return x

    first = record_func(store)(compute)
    second = record_func(store, name="OtherPlace.compute")(compute)

    first(1)
    second(1)
    assert len(records_for(store, first)) == 1
    assert len(records_for(store, second)) == 1
    assert first.identity != second.identity


def test_changed_source_warns_but_still_validates(store):
    def compute(x):
        return x * 2

    record_func(store)(compute)(5)

    def compute(x):  # noqa: F811 - same behavior, different source
        doubled = x * 2
        return doubled

    validating = record_func(store, mode=RunMode.VALIDATE_ONLY)(compute)
    with pytest.warns(CodeDriftWarning, match="has changed since this record"):
        assert validating(5) == 10
    assert validating.last_result.passed


def test_unchanged_source_does_not_warn(store, recwarn):
    recorded = record_func(store)(add)
    recorded(2, 3)
    recorded(2, 3)
    assert [w for w in recwarn.list if issubclass(w.category, CodeDriftWarning)] == []


def test_records_survive_a_new_wrapper_instance(store):
    record_func(store)(add)(2, 3)
    fresh = record_func(store, mode=RunMode.VALIDATE_ONLY)(add)
    assert fresh(2, 3) == 5
    assert fresh.last_result.passed
