import numpy as np
import pytest

from regression_records.config import OnNewCase, RunMode, resolve_on_new_case
from regression_records.exceptions import RecordNotFoundError, UnmatchedInvocationError
from regression_records.identity import FunctionIdentity
from regression_records.matching import (
    Action,
    Match,
    compute_input_hash,
    decide_action,
    find_match,
)
from regression_records.records import Invocation, RegressionRecord, SerializedValue
from regression_records.serialization import to_json_tree
from regression_records.store import RecordStore


def sample_function(a, b=1):
    return a + b


@pytest.fixture
def identity():
    return FunctionIdentity.from_function(sample_function)


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path)


def serialize(store, identity, args, kwargs, *, deferred=False):
    ctx = store.serialization_context(identity, deferred=deferred)
    return to_json_tree(list(args), ctx), to_json_tree(kwargs, ctx), ctx


def seed(store, identity, args, kwargs, output):
    """Record one invocation and return it."""
    args_tree, kwargs_tree, ctx = serialize(store, identity, args, kwargs)
    record = RegressionRecord.new(
        identity=identity,
        input_hash=compute_input_hash(args_tree, kwargs_tree),
        invocation=Invocation(args=SerializedValue(args_tree), kwargs=SerializedValue(kwargs_tree)),
        output=SerializedValue(to_json_tree(output, ctx)),
    )
    store.save(record)
    return record


# ------------------------------------------------------------------------ input_hash


def test_identical_inputs_hash_identically(store, identity):
    first = serialize(store, identity, (1, 2), {"c": 3})[:2]
    second = serialize(store, identity, (1, 2), {"c": 3})[:2]
    assert compute_input_hash(*first) == compute_input_hash(*second)


def test_different_inputs_hash_differently(store, identity):
    base = serialize(store, identity, (1, 2), {})[:2]
    assert compute_input_hash(*base) != compute_input_hash(
        *serialize(store, identity, (1, 3), {})[:2]
    )
    assert compute_input_hash(*base) != compute_input_hash(
        *serialize(store, identity, (1, 2), {"c": 1})[:2]
    )


def test_kwarg_order_does_not_affect_hash(store, identity):
    first = serialize(store, identity, (), {"a": 1, "b": 2})[:2]
    second = serialize(store, identity, (), {"b": 2, "a": 1})[:2]
    assert compute_input_hash(*first) == compute_input_hash(*second)


def test_positional_and_keyword_forms_hash_differently(store, identity):
    # Callers are expected to normalize via signature binding before hashing; the raw
    # trees genuinely differ, and matching does not guess.
    positional = serialize(store, identity, (1, 2), {})[:2]
    keyword = serialize(store, identity, (1,), {"b": 2})[:2]
    assert compute_input_hash(*positional) != compute_input_hash(*keyword)


def test_equal_arrays_hash_identically_across_objects(store, identity):
    first = serialize(store, identity, (np.arange(5),), {})[:2]
    second = serialize(store, identity, (np.arange(5),), {})[:2]
    assert compute_input_hash(*first) == compute_input_hash(*second)


def test_different_arrays_hash_differently(store, identity):
    first = serialize(store, identity, (np.arange(5),), {})[:2]
    second = serialize(store, identity, (np.arange(6),), {})[:2]
    assert compute_input_hash(*first) != compute_input_hash(*second)


def test_hash_is_stable_for_unordered_inputs(store, identity):
    # Sets serialize with deterministically sorted members, so they hash consistently.
    first = serialize(store, identity, ({3, 1, 2},), {})[:2]
    second = serialize(store, identity, ({2, 3, 1},), {})[:2]
    assert compute_input_hash(*first) == compute_input_hash(*second)


# ---------------------------------------------------------------------------- lookup


def test_find_match_locates_the_seeded_record(store, identity):
    recorded = seed(store, identity, (1, 2), {}, output=3)
    args_tree, kwargs_tree, _ = serialize(store, identity, (1, 2), {})

    match = find_match(store, identity, args_tree, kwargs_tree)
    assert not match.is_new_case
    assert match.record.record_id == recorded.record_id
    assert match.identity_has_records


def test_find_match_reports_a_new_case(store, identity):
    seed(store, identity, (1, 2), {}, output=3)
    args_tree, kwargs_tree, _ = serialize(store, identity, (9, 9), {})

    match = find_match(store, identity, args_tree, kwargs_tree)
    assert match.is_new_case
    assert match.record is None
    assert match.identity_has_records  # recorded before, just not with these inputs


def test_find_match_on_a_never_recorded_function(store, identity):
    args_tree, kwargs_tree, _ = serialize(store, identity, (1, 2), {})
    match = find_match(store, identity, args_tree, kwargs_tree)
    assert match.is_new_case
    assert not match.identity_has_records


def test_correct_record_selected_among_many(store, identity):
    expected = {}
    for value in range(10):
        expected[value] = seed(store, identity, (value,), {}, output=value * 2).record_id

    for value, record_id in expected.items():
        args_tree, kwargs_tree, _ = serialize(store, identity, (value,), {})
        match = find_match(store, identity, args_tree, kwargs_tree)
        assert match.record.record_id == record_id
        assert match.record.load_output(store.deserialization_context(identity)) == value * 2


def test_lookup_survives_an_array_argument(store, identity):
    seed(store, identity, (np.arange(4),), {}, output=6)
    args_tree, kwargs_tree, _ = serialize(store, identity, (np.arange(4),), {})
    assert not find_match(store, identity, args_tree, kwargs_tree).is_new_case


def test_input_hash_appears_in_the_record_filename(store, identity):
    record = seed(store, identity, (1, 2), {}, output=3)
    filenames = [p.name for p in store._records_dir(identity).glob("*.json")]
    assert filenames == [f"rec-{record.input_hash[:16]}-{record.record_id[:8]}.json"]


def test_lookup_does_not_parse_unrelated_records(store, identity, monkeypatch):
    for value in range(5):
        seed(store, identity, (value,), {}, output=value)
    args_tree, kwargs_tree, _ = serialize(store, identity, (3,), {})

    parsed = []
    original = RecordStore._load_record_file

    def counting_load(self, path):
        parsed.append(path)
        return original(self, path)

    monkeypatch.setattr(RecordStore, "_load_record_file", counting_load)
    assert not find_match(store, identity, args_tree, kwargs_tree).is_new_case
    assert len(parsed) == 1


# ------------------------------------------------------------------------- decisions


def new_case(has_records=True):
    return Match(input_hash="a" * 64, record=None, identity_has_records=has_records)


def matched():
    return Match(input_hash="a" * 64, record=object(), identity_has_records=True)


@pytest.mark.parametrize("on_new_case", list(OnNewCase))
def test_record_only_always_records(on_new_case):
    for match in (matched(), new_case(), new_case(has_records=False)):
        assert (
            decide_action(match, mode=RunMode.RECORD_ONLY, on_new_case=on_new_case)
            is Action.RECORD
        )


@pytest.mark.parametrize("mode", [RunMode.BOTH, RunMode.VALIDATE_ONLY])
@pytest.mark.parametrize("on_new_case", list(OnNewCase))
def test_a_match_always_validates(mode, on_new_case):
    assert decide_action(matched(), mode=mode, on_new_case=on_new_case) is Action.VALIDATE


def test_both_mode_records_a_new_case_by_default():
    assert (
        decide_action(new_case(), mode=RunMode.BOTH, on_new_case=OnNewCase.RECORD)
        is Action.RECORD
    )


def test_both_mode_ignores_a_new_case_when_asked():
    assert (
        decide_action(new_case(), mode=RunMode.BOTH, on_new_case=OnNewCase.IGNORE) is Action.SKIP
    )


def test_both_mode_errors_on_a_new_case_when_asked():
    with pytest.raises(UnmatchedInvocationError, match="OnNewCase.ERROR"):
        decide_action(new_case(), mode=RunMode.BOTH, on_new_case=OnNewCase.ERROR)


def test_validate_only_without_any_records_raises():
    with pytest.raises(RecordNotFoundError, match="record a run first"):
        decide_action(
            new_case(has_records=False), mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.ERROR
        )


def test_validate_only_surfaces_a_new_case_and_names_the_mode():
    # ERROR is what resolve_on_new_case() hands down for VALIDATE_ONLY, and the message
    # says so, since the user may never have chosen it themselves.
    with pytest.raises(UnmatchedInvocationError, match="VALIDATE_ONLY"):
        decide_action(new_case(), mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.ERROR)


def test_an_explicitly_chosen_error_does_not_blame_the_mode():
    with pytest.raises(UnmatchedInvocationError) as excinfo:
        decide_action(new_case(), mode=RunMode.BOTH, on_new_case=OnNewCase.ERROR)
    assert "VALIDATE_ONLY" not in str(excinfo.value)


def test_validate_only_can_ignore_a_new_case():
    assert (
        decide_action(new_case(), mode=RunMode.VALIDATE_ONLY, on_new_case=OnNewCase.IGNORE)
        is Action.SKIP
    )


# ---------------------------------------------------------------- resolving the default


@pytest.mark.parametrize("mode", [RunMode.RECORD_ONLY, RunMode.BOTH])
def test_unset_on_new_case_records_where_recording_is_allowed(mode):
    assert resolve_on_new_case(mode, None) is OnNewCase.RECORD


def test_unset_on_new_case_errors_under_validate_only():
    assert resolve_on_new_case(RunMode.VALIDATE_ONLY, None) is OnNewCase.ERROR


@pytest.mark.parametrize("mode", list(RunMode))
@pytest.mark.parametrize("on_new_case", list(OnNewCase))
def test_an_explicit_choice_is_kept_unless_it_contradicts_the_mode(mode, on_new_case):
    contradictory = mode is RunMode.VALIDATE_ONLY and on_new_case is OnNewCase.RECORD
    if contradictory:
        with pytest.raises(ValueError, match="contradicts"):
            resolve_on_new_case(mode, on_new_case)
    else:
        assert resolve_on_new_case(mode, on_new_case) is on_new_case


# -------------------------------------------------------------- deferred arg writing


def test_deferred_context_writes_nothing_until_flushed(store, identity):
    ctx = store.serialization_context(identity, deferred=True)
    to_json_tree({"arr": np.arange(10)}, ctx)

    data_dir = store._data_dir(identity)
    assert not data_dir.exists()
    assert ctx.pending_writes == 1

    written = ctx.flush()
    assert len(written) == 1
    assert len(list(data_dir.glob("*.npy"))) == 1
    assert ctx.pending_writes == 0


def test_discarded_deferred_writes_leave_no_files(store, identity):
    ctx = store.serialization_context(identity, deferred=True)
    to_json_tree({"arr": np.arange(10)}, ctx)
    ctx.discard()
    ctx.flush()

    assert not any(store._data_dir(identity).glob("*.npy"))
    assert ctx.pending_writes == 0


def test_deferred_and_eager_serialization_agree(store, identity):
    eager_args, eager_kwargs, _ = serialize(store, identity, (np.arange(5),), {})
    deferred_args, deferred_kwargs, ctx = serialize(
        store, identity, (np.arange(5),), {}, deferred=True
    )
    assert compute_input_hash(eager_args, eager_kwargs) == compute_input_hash(
        deferred_args, deferred_kwargs
    )
    ctx.discard()


def test_deferred_flush_is_idempotent_for_identical_content(store, identity):
    ctx = store.serialization_context(identity, deferred=True)
    to_json_tree([np.arange(3), np.arange(3)], ctx)
    assert ctx.pending_writes == 1  # content-addressed, so both refer to one file
    ctx.flush()

    again = store.serialization_context(identity, deferred=True)
    to_json_tree([np.arange(3)], again)
    assert again.flush() == []  # already on disk


def test_a_validating_lookup_leaves_no_orphan_files(store, identity):
    seed(store, identity, (np.arange(4),), {}, output=1)
    files_after_seeding = set(p.name for p in store._data_dir(identity).iterdir())

    # A validate-style pass over inputs that do not match must not grow the store.
    args_tree, kwargs_tree, ctx = serialize(store, identity, (np.arange(99),), {}, deferred=True)
    match = find_match(store, identity, args_tree, kwargs_tree)
    assert match.is_new_case
    ctx.discard()

    assert set(p.name for p in store._data_dir(identity).iterdir()) == files_after_seeding
