import pytest

from regression_records._engine import forget_recorded_calls
from regression_records.serialization import (
    DeserializationContext,
    SerializationContext,
    default_registry,
)

# replay_tests is tested by running pytest on a generated test module.
pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def recording_on(monkeypatch):
    """Switch recording on, as a fresh process would find it.

    Most tests exercise recording, which is off unless ``REGRESSION_RECORDS`` says
    otherwise; the tests of the switch itself unset it. The per-process memory of
    recorded calls is cleared so that no test sees another's.
    """
    monkeypatch.setenv("REGRESSION_RECORDS", "1")
    forget_recorded_calls()


@pytest.fixture()
def ctx_pair(tmp_path):
    """A (SerializationContext, DeserializationContext) pair sharing a directory."""
    ser_ctx = SerializationContext(data_dir=tmp_path, registry=default_registry)
    deser_ctx = DeserializationContext(base_dir=tmp_path, registry=default_registry)
    return ser_ctx, deser_ctx
