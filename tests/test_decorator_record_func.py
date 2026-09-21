import os
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

from regression_records import disable, enable, recording
from regression_records._engine import forget_recorded_calls
from regression_records.decorators import record_func
from regression_records.exceptions import SerializationError
from regression_records.store import RecordStore


def add(a, b=1, *, scale=1):
    """Add two numbers, then scale."""
    return (a + b) * scale


def boom(x):
    raise ValueError(f"bad {x}")


def interrupted(x):
    raise KeyboardInterrupt


def arange(n):
    return np.arange(n)


def make_lambda(x):
    return lambda: x


def passthrough(value):
    return value


def returns_none(x):
    return None


#: What ``offset_add`` adds; a test changes it to change the function's behavior.
OFFSET = 0


def offset_add(a, b):
    return a + b + OFFSET


#: The decorated function ``outer`` calls; a test installs a recorder here.
call_inner = make_lambda


def outer(x):
    return call_inner(x)


def transform(frame, weights, meta):
    return {
        "frame": frame.with_columns(pl.col("a") * 2),
        "total": weights.sum(),
        "meta": meta,
    }


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
    with pytest.raises(ValueError, match="bad 1"):
        record_func(store)(boom)(1)


def test_wrong_arguments_still_raise_the_real_type_error(store):
    recorded = record_func(store)(add)
    with pytest.raises(TypeError):
        recorded(1, 2, 3, 4)
    assert records_for(store, recorded) == []


def test_store_accepts_a_path(tmp_path):
    recorded = record_func(tmp_path / "records")(add)
    assert recorded(1, 1) == 2
    assert (tmp_path / "records").exists()


def test_the_wrapper_has_no_result_slot(store):
    recorded = record_func(store)(add)
    recorded(1)
    assert not hasattr(recorded, "last_result")
    assert not hasattr(recorded, "config")


# ----------------------------------------------------------------- the switch


def test_recording_is_off_unless_switched_on(store, monkeypatch):
    monkeypatch.delenv("REGRESSION_RECORDS")
    recorded = record_func(store)(add)

    assert recorded(2, 3) == 5
    assert not store.root.joinpath(*recorded.identity.path_parts).exists()


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_environment_switches_recording_on(store, monkeypatch, value):
    monkeypatch.setenv("REGRESSION_RECORDS", value)
    recorded = record_func(store)(add)
    recorded(2, 3)
    assert len(records_for(store, recorded)) == 1


@pytest.mark.parametrize("value", ["", "0", "false", "No", "off"])
def test_the_environment_switches_recording_off(store, monkeypatch, value):
    monkeypatch.setenv("REGRESSION_RECORDS", value)
    recorded = record_func(store)(add)
    recorded(2, 3)
    assert records_for(store, recorded) == []


def test_an_unrecognised_switch_value_is_an_error(store, monkeypatch):
    monkeypatch.setenv("REGRESSION_RECORDS", "yes please")
    with pytest.raises(ValueError, match="not a recognised switch"):
        record_func(store)(add)(2, 3)


def test_the_switch_is_read_on_every_call(store, monkeypatch):
    monkeypatch.delenv("REGRESSION_RECORDS")
    recorded = record_func(store)(add)
    recorded(1, 1)
    enable()
    recorded(2, 2)
    disable()
    recorded(3, 3)

    (written,) = records_for(store, recorded)
    assert written.load_args(store.deserialization_context(recorded.identity)) == [2, 2]


@pytest.mark.parametrize("enabled", [True, False])
def test_recording_block_sets_and_restores_the_switch(store, monkeypatch, enabled):
    monkeypatch.setenv("REGRESSION_RECORDS", "sentinel-restored")
    recorded = record_func(store)(add)
    with recording(enabled):
        recorded(2, 3)

    assert len(records_for(store, recorded)) == int(enabled)
    assert os.environ["REGRESSION_RECORDS"] == "sentinel-restored"


def test_enabled_false_is_a_hard_off(tmp_path):
    store = RecordStore(tmp_path / "store")
    recorded = record_func(store, enabled=False)(add)
    assert recorded(2, 3) == 5
    assert not (tmp_path / "store").exists()


# --------------------------------------------------------------------- recording


def test_a_call_records_its_arguments_and_result(store):
    recorded = record_func(store)(add)
    recorded(2, 3)

    (written,) = records_for(store, recorded)
    ctx = store.deserialization_context(recorded.identity)
    assert written.load_output(ctx) == 5
    assert written.load_args(ctx) == [2, 3]
    assert written.load_kwargs(ctx) == {"scale": 1}


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


def test_a_call_repeated_in_one_process_is_recorded_once(store, monkeypatch):
    recorded = record_func(store)(add)
    recorded(2, 3)
    saves = []
    monkeypatch.setattr(store, "save", lambda record, **kw: saves.append(record))

    for _ in range(5):
        recorded(2, 3)

    assert saves == []


def test_recording_the_same_behavior_again_writes_nothing_new(store):
    recorded = record_func(store)(add)
    recorded(2, 3)
    first = records_for(store, recorded)

    forget_recorded_calls()  # as a new process would
    recorded(2, 3)

    assert records_for(store, recorded) == first


def test_changed_behavior_is_recorded_beside_the_old_record(store, monkeypatch):
    """Records are never replaced: removing the old one is a manual act."""
    recorded = record_func(store)(offset_add)
    recorded(2, 3)

    monkeypatch.setattr(sys.modules[__name__], "OFFSET", 1)
    forget_recorded_calls()
    recorded(2, 3)

    written = records_for(store, recorded)
    ctx = store.deserialization_context(recorded.identity)
    assert sorted(r.load_output(ctx) for r in written) == [5, 6]
    assert len({r.input_hash for r in written}) == 1


def test_redefining_a_function_records_again_in_the_same_process(store):
    """An intended change made in a notebook is recorded without restarting it."""
    record_func(store)(add)(2, 3)

    def add_changed(a, b=1, *, scale=1):
        return (a + b) * scale + 1

    add_changed.__qualname__ = add.__qualname__
    add_changed.__module__ = add.__module__
    recorded = record_func(store)(add_changed)
    recorded(2, 3)

    ctx = store.deserialization_context(recorded.identity)
    assert sorted(r.load_output(ctx) for r in records_for(store, recorded)) == [5, 6]


def test_parallel_recorders_of_the_same_behavior_converge_on_one_file(tmp_path):
    """Workers of a pytest-xdist run each record the same case (#5)."""
    (tmp_path / "worker_target.py").write_text("def add(a, b):\n    return a + b\n")
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[2])
        from regression_records import record_func
        from worker_target import add

        record_func(sys.argv[1])(add)(2, 3)
        """
    )
    store = tmp_path / "store"
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(store), str(tmp_path)])
        for _ in range(4)
    ]
    assert [w.wait() for w in workers] == [0] * 4

    assert len(list(store.rglob("rec-*.json"))) == 1


def test_tags_are_stored(store):
    recorded = record_func(store, tags={"suite": "smoke"})(add)
    recorded(1, 1)
    assert records_for(store, recorded)[0].tags == {"suite": "smoke"}


def test_complex_arguments_and_return_values(store):
    recorded = record_func(store)(transform)
    args = (pl.DataFrame({"a": [1, 2]}), np.array([1.0, 2.0]), {"n": (1, 2)})
    result = recorded(*args)

    (written,) = records_for(store, recorded)
    output = written.load_output(store.deserialization_context(recorded.identity))
    assert output["total"] == result["total"]
    assert output["frame"].equals(result["frame"])


def test_a_frame_argument_and_the_same_frame_as_output_share_one_file(store):
    recorded = record_func(store)(passthrough)
    recorded(pl.DataFrame({"a": [1, 2, 3]}))
    assert len(list(store.root.rglob("*.parquet"))) == 1


def test_none_return_value_is_recorded(store):
    recorded = record_func(store)(returns_none)
    assert recorded(1) is None
    assert records_for(store, recorded)[0].succeeded


# --------------------------------------------------------------------- exceptions


def test_exceptions_are_recorded_by_default(store):
    recorded = record_func(store)(boom)
    with pytest.raises(ValueError):
        recorded(1)

    (stored,) = records_for(store, recorded)
    assert not stored.succeeded
    assert stored.exception.exc_type == "builtins.ValueError"
    assert stored.exception.message == "bad 1"
    assert "raise ValueError" in stored.exception.traceback_text


def test_record_exceptions_false_records_nothing_on_failure(store):
    recorded = record_func(store, record_exceptions=False)(boom)
    with pytest.raises(ValueError):
        recorded(1)
    assert records_for(store, recorded) == []


def test_keyboard_interrupt_propagates_without_being_recorded(store):
    recorded = record_func(store)(interrupted)
    with pytest.raises(KeyboardInterrupt):
        recorded(1)
    assert records_for(store, recorded) == []


def test_unserializable_argument_raises(store):
    recorded = record_func(store)(add)
    with pytest.raises(SerializationError):
        recorded(object, 1)


def test_unserializable_output_raises(store):
    recorded = record_func(store)(make_lambda)
    with pytest.raises(SerializationError):
        recorded(1)
    assert records_for(store, recorded) == []


def test_a_nested_recording_failure_is_not_recorded_as_the_outer_behaviour(
    store, monkeypatch
):
    """A library error from an inner call propagates; it is not the outer's result."""
    monkeypatch.setattr(
        sys.modules[__name__], "call_inner", record_func(store)(make_lambda)
    )
    recorded_outer = record_func(store)(outer)
    with pytest.raises(SerializationError):
        recorded_outer(1)
    assert records_for(store, recorded_outer) == []


# ------------------------------------------------------------- write ordering


def test_side_files_are_written_before_the_record_that_references_them(
    store, monkeypatch
):
    """An interrupted save must not leave a record pointing at missing data."""

    def failing_save(record, *, overwrite=False):
        raise OSError("disk full")

    recorded = record_func(store)(arange)
    monkeypatch.setattr(store, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        recorded(4)

    data_dir = store.root.joinpath(*recorded.identity.path_parts, "data")
    assert [p.suffix for p in data_dir.iterdir()] == [".npy"]


def test_an_interrupted_write_leaves_no_file_under_the_final_name(store, monkeypatch):
    def failing_save(path, array, **kwargs):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(np, "save", failing_save)
    recorded = record_func(store)(arange)
    with pytest.raises(OSError, match="interrupted"):
        recorded(4)

    data_dir = store.root.joinpath(*recorded.identity.path_parts, "data")
    assert list(data_dir.iterdir()) == []


def test_a_different_record_under_the_same_filename_is_not_lost(store, monkeypatch):
    """Record ids share an 8-character filename prefix; a clash must not be silent."""
    recorded = record_func(store)(add)
    recorded(2, 3)
    (path,) = store.iter_record_paths()
    text = path.read_text()
    (existing,) = records_for(store, recorded)
    path.write_text(text.replace(existing.record_id, existing.record_id[:8] + "0" * 24))

    forget_recorded_calls()
    with pytest.raises(SerializationError, match="taken by a different record"):
        recorded(2, 3)


# --------------------------------------------------------- unrecordable callables


def test_generator_function_is_rejected_at_decoration(store):
    def numbers():
        yield 1

    with pytest.raises(TypeError, match="generator"):
        record_func(store)(numbers)


def test_async_function_is_rejected_at_decoration(store):
    async def fetch():
        return 1

    with pytest.raises(TypeError, match="async"):
        record_func(store)(fetch)


def test_a_class_passed_as_a_value_is_refused_cleanly(store):
    """``SomeClass.__getstate__()`` is unbound, and used to escape as a TypeError."""
    with pytest.raises(SerializationError, match="custom Serializer"):
        record_func(store)(passthrough)(RecordStore)
