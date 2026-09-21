import pytest

from regression_records.serialization import (
    DeserializationContext,
    SerializationContext,
    default_registry,
)

# replay_tests is tested by running pytest on a generated test module.
pytest_plugins = ["pytester"]


@pytest.fixture()
def ctx_pair(tmp_path):
    """A (SerializationContext, DeserializationContext) pair sharing a directory."""
    ser_ctx = SerializationContext(data_dir=tmp_path, registry=default_registry)
    deser_ctx = DeserializationContext(base_dir=tmp_path, registry=default_registry)
    return ser_ctx, deser_ctx
