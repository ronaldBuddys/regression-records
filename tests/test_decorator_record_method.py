import tempfile
import threading
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from regression_records.decorators import record_method
from regression_records.exceptions import SerializationError
from regression_records.store import RecordStore

#: The store every class below records to. Classes must be defined at module level to
#: be recorded, so each test points this one store at its own directory instead.
STORE = RecordStore(Path(tempfile.gettempdir()) / "regression-records-never-written")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(STORE, "root", tmp_path)
    return STORE


class Counter:
    def __init__(self, n=0, label="c"):
        self.n = n
        self.label = label

    @record_method(STORE)
    def bump(self, by=1):
        self.n += by
        return self.n

    @record_method(STORE)
    def explode(self):
        self.n = 99
        raise ValueError("nope")


class DisabledCounter(Counter):
    @record_method(STORE, enabled=False)
    def bump(self, by=1):
        self.n += by
        return self.n


class OnlyN(Counter):
    @record_method(STORE, state_attrs=["n"], factory=Counter)
    def bump(self, by=1):
        self.n += by
        return self.n


class WithoutLabel(Counter):
    @record_method(STORE, exclude_attrs=["label"], factory=Counter)
    def bump(self, by=1):
        self.n += by
        return self.n


class Misspelt(Counter):
    @record_method(STORE, state_attrs=["count"], factory=Counter)
    def bump(self, by=1):
        self.n += by
        return self.n


class Bag:
    def __init__(self):
        self.items = [1, 2]

    @record_method(STORE)
    def add(self, x):
        self.items.append(x)
        return len(self.items)


class Reader:
    def __init__(self, n):
        self.n = n

    @record_method(STORE)
    def doubled(self):
        return self.n * 2


class Uncopyable:
    def __deepcopy__(self, memo):
        raise TypeError("this object refuses to be copied")


class Client:
    def __init__(self):
        self.conn = Uncopyable()
        self.n = 0

    @record_method(STORE, exclude_attrs=["conn"], factory=lambda: Client())
    def bump(self):
        self.n += 1
        return self.n

    @record_method(STORE)
    def touch(self):
        return 1


class Slotted:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0

    @record_method(STORE)
    def bump(self):
        self.n += 1
        return self.n


class Model:
    def __init__(self, weights=None):
        self.weights = weights

    @record_method(STORE)
    def scale(self, factor):
        self.weights = self.weights * factor
        return float(self.weights.sum())


class Fitter:
    @record_method(STORE, state_attrs=["coef_"], factory=lambda: Fitter())
    def fit(self, x):
        self.coef_ = x * 2


class Table:
    def __init__(self, frame):
        self.frame = frame

    @record_method(STORE)
    def drop_first(self):
        self.frame = self.frame.slice(1)
        return self.frame.height


class Engine:
    def __init__(self, rpm):
        self.rpm = rpm


class Car:
    def __init__(self, engine):
        self.engine = engine

    @record_method(STORE)
    def rev(self, delta):
        self.engine.rpm += delta
        return self.engine.rpm


class Worker:
    def __init__(self):
        self.done = 0

    @record_method(STORE)
    def run(self):
        self.lock = threading.Lock()  # cannot be deep-copied
        raise RuntimeError("failed part-way")


def records_for(store, method):
    return store.list_records(method.identity)


def only_record(store, method):
    (record,) = records_for(store, method)
    return record


# ------------------------------------------------------------------ transparency


def test_return_value_and_mutation_pass_through(store):
    obj = Counter(n=10)
    assert obj.bump(5) == 15
    assert obj.n == 15


def test_binding_semantics_are_unaffected(store):
    # Bound call, unbound call, and getattr-then-call must all still work.
    assert Counter(n=1).bump(1) == 2
    assert Counter.bump(Counter(n=1), 2) == 3
    assert Counter(n=1).bump(3) == 4
    assert Counter.bump.__name__ == "bump"


def test_disabled_decorator_records_nothing(store):
    assert DisabledCounter(n=1).bump() == 2
    assert records_for(store, DisabledCounter.bump) == []


def test_exception_propagates_and_is_recorded(store):
    obj = Counter(n=1)
    with pytest.raises(ValueError, match="nope"):
        obj.explode()
    assert obj.n == 99
    record = only_record(store, Counter.explode)
    assert record.exception.exc_type == "builtins.ValueError"
    # State after a failed call still matters: the object was mutated before it raised.
    assert record.state_after.data["n"] == 99


# ------------------------------------------------------------------------ recording


def test_self_is_excluded_from_recorded_arguments(store):
    Counter(n=3).bump(2)
    record = only_record(store, Counter.bump)
    assert record.invocation.args.data == []
    assert record.invocation.kwargs.data == {"by": 2}


def test_state_before_and_after_are_both_recorded(store):
    Counter(n=3).bump(4)
    record = only_record(store, Counter.bump)
    assert record.state_before.data == {"n": 3, "label": "c"}
    assert record.state_after.data == {"n": 7, "label": "c"}


def test_state_before_is_snapshotted_not_referenced(store):
    """The whole point of deep-copying: a mutated container must not rewrite history."""
    Bag().add(3)
    record = only_record(store, Bag.add)
    assert record.state_before.data == {"items": [1, 2]}
    assert record.state_after.data == {"items": [1, 2, 3]}


def test_a_method_that_mutates_nothing_records_equal_states(store):
    assert Reader(4).doubled() == 8
    record = only_record(store, Reader.doubled)
    assert record.state_before.data == record.state_after.data == {"n": 4}


def test_instances_in_different_states_are_recorded_separately(store):
    """``self`` is not hashed, but its state is part of what a record describes."""
    assert Counter(n=1).bump(1) == 2
    assert Counter(n=100).bump(1) == 101
    records = records_for(store, Counter.bump)
    assert sorted(r.state_before.data["n"] for r in records) == [1, 100]
    assert len({r.input_hash for r in records}) == 1


def test_no_cross_instance_contamination(store):
    first, second = Counter(n=0), Counter(n=0)
    assert first.bump(1) == 1
    assert second.bump(1) == 1
    assert (first.n, second.n) == (1, 1)
    assert len(records_for(store, Counter.bump)) == 1


# --------------------------------------------------------------- attribute selection


def test_state_attrs_records_only_the_listed_attributes(store):
    OnlyN(n=3, label="ignored").bump(1)
    assert only_record(store, OnlyN.bump).state_before.data == {"n": 3}


def test_exclude_attrs_leaves_an_attribute_out(store):
    WithoutLabel(n=3).bump(1)
    assert only_record(store, WithoutLabel.bump).state_before.data == {"n": 3}


@pytest.mark.parametrize(
    "selection", [{"state_attrs": ["n"]}, {"exclude_attrs": ["label"]}]
)
def test_a_partial_state_needs_a_factory(selection):
    """Replay cannot rebuild an object from part of its state on its own."""
    with pytest.raises(TypeError, match="factory="):
        record_method(STORE, **selection)


def test_a_factory_must_be_callable():
    with pytest.raises(TypeError, match="factory must be callable"):
        record_method(STORE, factory=Counter())


def test_exclude_attrs_is_the_escape_hatch_for_uncopyable_state(store):
    assert Client().bump() == 1
    assert only_record(store, Client.bump).state_after.data == {"n": 1}


def test_uncopyable_state_without_an_exclusion_reports_the_escape_hatch(store):
    with pytest.raises(SerializationError, match="exclude_attrs"):
        Client().touch()


def test_slots_only_receiver_is_rejected_clearly(store):
    with pytest.raises(SerializationError, match="__slots__"):
        Slotted().bump()


# ------------------------------------------------------------------- rich attributes


def test_numpy_attribute_round_trips(store):
    Model(np.array([1.0, 2.0, 3.0])).scale(2.0)

    record = only_record(store, Model.scale)
    ctx = store.deserialization_context(Model.scale.identity)
    np.testing.assert_array_equal(
        record.load_state_after(ctx)["weights"], [2.0, 4.0, 6.0]
    )


def test_polars_attribute_round_trips(store):
    Table(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})).drop_first()

    record = only_record(store, Table.drop_first)
    ctx = store.deserialization_context(Table.drop_first.identity)
    assert record.load_state_after(ctx)["frame"].equals(
        pl.DataFrame({"a": [2, 3], "b": ["y", "z"]})
    )


def test_nested_custom_object_state_is_recorded_attribute_by_attribute(store):
    Car(Engine(1000)).rev(500)
    record = only_record(store, Car.rev)
    assert record.state_after.data["engine"]["state"]["rpm"] == 1500


# ------------------------------------------------------------- state selection


def test_a_misspelt_state_attr_is_an_error_not_an_empty_state(store):
    """A typo used to select nothing, so the method was tested against ``{}``."""
    with pytest.raises(SerializationError, match=r"state_attrs names \['count'\]"):
        Misspelt().bump()


def test_a_state_attr_the_method_creates_is_not_an_error(store):
    Fitter().fit(3)
    record = only_record(store, Fitter.fit)
    assert record.state_before.data == {}
    assert record.state_after.data == {"coef_": 6}


# ------------------------------------------------------- after-state not captured


def test_an_uncapturable_after_state_is_recorded_as_absent_not_null(store):
    """``state_after: null`` could never match a later run's real state."""
    with pytest.raises(RuntimeError, match="part-way"):
        Worker().run()
    record = only_record(store, Worker.run)
    assert record.state_before.data == {"done": 0}
    assert record.state_after is None


# --------------------------------------------------------- unrecordable callables


def _plain(*args):
    return None


@pytest.mark.parametrize("wrapper", [staticmethod, classmethod])
def test_decorating_above_staticmethod_or_classmethod_is_rejected(store, wrapper):
    with pytest.raises(TypeError, match=f"beneath @{wrapper.__name__}"):
        record_method(store)(wrapper(_plain))


def test_a_method_of_a_class_defined_in_a_function_is_refused(store):
    with pytest.raises(TypeError, match="defined inside a function"):

        class Local:
            @record_method(store)
            def go(self):
                return 1
