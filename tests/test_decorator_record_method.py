import numpy as np
import polars as pl
import pytest

from regression_records.decorators import record_method
from regression_records.exceptions import SerializationError
from regression_records.store import RecordStore


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path)


def counter_class(decorator):
    """A small mutating class whose method is wrapped by ``decorator``.

    Built by a factory so each test gets a fresh class, while every class the factory
    builds shares one qualname - and therefore one identity.
    """

    class Counter:
        def __init__(self, n=0, label="c"):
            self.n = n
            self.label = label

        @decorator
        def bump(self, by=1):
            self.n += by
            return self.n

    return Counter


def records_for(store, cls):
    return store.list_records(cls.bump.identity)


def only_record(store, cls):
    records = records_for(store, cls)
    assert len(records) == 1
    return records[0]


# ------------------------------------------------------------------ transparency


def test_return_value_and_mutation_pass_through(store):
    obj = counter_class(record_method(store))(n=10)
    assert obj.bump(5) == 15
    assert obj.n == 15


def test_binding_semantics_are_unaffected(store):
    cls = counter_class(record_method(store))
    # Bound call, unbound call, and getattr-then-call must all still work. Each uses a
    # distinct argument so each is its own recorded case rather than a re-validation.
    assert cls(n=1).bump(1) == 2
    assert cls.bump(cls(n=1), 2) == 3
    assert cls(n=1).bump(3) == 4
    assert cls.bump.__name__ == "bump"


def test_disabled_decorator_records_nothing(store):
    cls = counter_class(record_method(store, enabled=False))
    assert cls(n=1).bump() == 2
    assert records_for(store, cls) == []


def test_exception_propagates_and_is_recorded(store):
    def boom(self):
        self.n = 99
        raise ValueError("nope")

    cls = counter_class(record_method(store))
    cls.explode = record_method(store)(boom)
    obj = cls(n=1)
    with pytest.raises(ValueError, match="nope"):
        obj.explode()
    assert obj.n == 99
    records = store.list_records(cls.explode.identity)
    assert len(records) == 1
    assert records[0].exception.exc_type == "builtins.ValueError"
    # State after a failed call still matters: the object was mutated before it raised.
    assert records[0].state_after.data["n"] == 99


# ------------------------------------------------------------------------ recording


def test_self_is_excluded_from_recorded_arguments(store):
    cls = counter_class(record_method(store))
    cls(n=3).bump(2)
    record = only_record(store, cls)
    assert record.invocation.args.data == []
    assert record.invocation.kwargs.data == {"by": 2}
    assert "self" not in record.invocation.kwargs.data


def test_state_before_and_after_are_both_recorded(store):
    cls = counter_class(record_method(store))
    cls(n=3).bump(4)
    record = only_record(store, cls)
    assert record.state_before.data == {"n": 3, "label": "c"}
    assert record.state_after.data == {"n": 7, "label": "c"}


def test_state_before_is_snapshotted_not_referenced(store):
    """The whole point of deep-copying: a mutated container must not rewrite history."""

    class Bag:
        def __init__(self):
            self.items = [1, 2]

        @record_method(store)
        def add(self, x):
            self.items.append(x)
            return len(self.items)

    Bag().add(3)
    record = store.list_records(Bag.add.identity)[0]
    assert record.state_before.data == {"items": [1, 2]}
    assert record.state_after.data == {"items": [1, 2, 3]}


def test_a_method_that_mutates_nothing_records_equal_states(store):
    class Reader:
        def __init__(self, n):
            self.n = n

        @record_method(store)
        def doubled(self):
            return self.n * 2

    assert Reader(4).doubled() == 8
    record = store.list_records(Reader.doubled.identity)[0]
    assert record.state_before.data == record.state_after.data == {"n": 4}


def test_instances_in_different_states_are_recorded_separately(store):
    """``self`` is not hashed, but its state is part of what a record describes."""
    cls = counter_class(record_method(store))
    assert cls(n=1).bump(1) == 2
    assert cls(n=100).bump(1) == 101
    records = records_for(store, cls)
    assert sorted(r.state_before.data["n"] for r in records) == [1, 100]
    assert len({r.input_hash for r in records}) == 1


def test_no_cross_instance_contamination(store):
    cls = counter_class(record_method(store))
    first, second = cls(n=0), cls(n=0)
    assert first.bump(1) == 1
    assert second.bump(1) == 1
    assert (first.n, second.n) == (1, 1)
    assert len(records_for(store, cls)) == 1


# --------------------------------------------------------------- attribute selection


def test_state_attrs_records_only_the_listed_attributes(store):
    cls = counter_class(record_method(store, state_attrs=["n"]))
    cls(n=3, label="ignored").bump(1)
    record = only_record(store, cls)
    assert record.state_before.data == {"n": 3}


def test_exclude_attrs_leaves_an_attribute_out(store):
    cls = counter_class(record_method(store, exclude_attrs=["label"]))
    cls(n=3).bump(1)
    record = only_record(store, cls)
    assert record.state_before.data == {"n": 3}


def test_exclude_attrs_is_the_escape_hatch_for_uncopyable_state(store):
    class Client:
        def __init__(self):
            self.conn = _Uncopyable()
            self.n = 0

        @record_method(store, exclude_attrs=["conn"])
        def bump(self):
            self.n += 1
            return self.n

    assert Client().bump() == 1
    assert store.list_records(Client.bump.identity)[0].state_after.data == {"n": 1}


def test_uncopyable_state_without_an_exclusion_reports_the_escape_hatch(store):
    class Client:
        def __init__(self):
            self.conn = _Uncopyable()

        @record_method(store)
        def touch(self):
            return 1

    with pytest.raises(SerializationError, match="exclude_attrs"):
        Client().touch()


def test_slots_only_receiver_is_rejected_clearly(store):
    class Slotted:
        __slots__ = ("n",)

        def __init__(self):
            self.n = 0

        @record_method(store)
        def bump(self):
            self.n += 1
            return self.n

    with pytest.raises(SerializationError, match="__slots__"):
        Slotted().bump()


class _Uncopyable:
    def __deepcopy__(self, memo):
        raise TypeError("this object refuses to be copied")


# ------------------------------------------------------------------- rich attributes


def test_numpy_attribute_round_trips(store):
    class Model:
        def __init__(self, weights):
            self.weights = weights

        @record_method(store)
        def scale(self, factor):
            self.weights = self.weights * factor
            return float(self.weights.sum())

    Model(np.array([1.0, 2.0, 3.0])).scale(2.0)

    (record,) = store.list_records(Model.scale.identity)
    ctx = store.deserialization_context(Model.scale.identity)
    np.testing.assert_array_equal(
        record.load_state_after(ctx)["weights"], [2.0, 4.0, 6.0]
    )


def test_polars_attribute_round_trips(store):
    class Table:
        def __init__(self, frame):
            self.frame = frame

        @record_method(store)
        def drop_first(self):
            self.frame = self.frame.slice(1)
            return self.frame.height

    Table(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})).drop_first()

    (record,) = store.list_records(Table.drop_first.identity)
    ctx = store.deserialization_context(Table.drop_first.identity)
    assert record.load_state_after(ctx)["frame"].equals(
        pl.DataFrame({"a": [2, 3], "b": ["y", "z"]})
    )


def test_nested_custom_object_state_is_compared_attribute_by_attribute(store):
    class Engine:
        def __init__(self, rpm):
            self.rpm = rpm

    class Car:
        def __init__(self, engine):
            self.engine = engine

        @record_method(store)
        def rev(self, delta):
            self.engine.rpm += delta
            return self.engine.rpm

    # A nested object must be importable to be recorded, which locals aren't.
    Engine.__module__ = Car.__module__ = __name__
    Engine.__qualname__ = "_Engine"
    globals()["_Engine"] = Engine

    Car(Engine(1000)).rev(500)
    record = store.list_records(Car.rev.identity)[0]
    assert record.state_after.data["engine"]["state"]["rpm"] == 1500


# ------------------------------------------------------------- state selection


def test_a_misspelt_state_attr_is_an_error_not_an_empty_state(store):
    """A typo used to select nothing, so the method was tested against ``{}``."""
    Counter = counter_class(record_method(store, state_attrs=["count"]))
    with pytest.raises(SerializationError, match=r"state_attrs names \['count'\]"):
        Counter().bump()


def test_a_state_attr_the_method_creates_is_not_an_error(store):
    class Model:
        @record_method(store, state_attrs=["coef_"])
        def fit(self, x):
            self.coef_ = x * 2

    Model().fit(3)
    record = store.list_records(Model.fit.identity)[0]
    assert record.state_before.data == {}
    assert record.state_after.data == {"coef_": 6}


# ------------------------------------------------------- after-state not captured


def test_an_uncapturable_after_state_is_recorded_as_absent_not_null(store):
    """``state_after: null`` could never match a later run's real state."""
    import threading

    class Worker:
        def __init__(self):
            self.done = 0

        @record_method(store)
        def run(self):
            self.lock = threading.Lock()  # cannot be deep-copied
            raise RuntimeError("failed part-way")

    with pytest.raises(RuntimeError, match="part-way"):
        Worker().run()
    record = store.list_records(Worker.run.identity)[0]
    assert record.state_before.data == {"done": 0}
    assert record.state_after is None


# --------------------------------------------------------- unrecordable callables


@pytest.mark.parametrize("wrapper", [staticmethod, classmethod])
def test_decorating_above_staticmethod_or_classmethod_is_rejected(store, wrapper):
    with pytest.raises(TypeError, match=f"beneath @{wrapper.__name__}"):
        record_method(store)(wrapper(lambda *args: None))
