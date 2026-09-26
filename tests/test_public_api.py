"""The package's public surface.

``__all__`` is a promise, so it is worth a test: a name that appears there becomes
something users import and something this project has to keep. These tests exist to make
adding or removing one a deliberate act rather than a side effect of an import edit.
"""

import importlib

import pytest

import regression_records as rr

EXPECTED_API = {
    # decorators
    "record_func",
    "record_method",
    # replay
    "replay",
    "replay_tests",
    "ReplayCase",
    "ReplayReport",
    "ReplayStatus",
    # the recording switch
    "enable",
    "disable",
    "recording",
    # run manifests
    "session",
    "environment",
    "Environment",
    "RecordingSession",
    # configuration
    "ToleranceConfig",
    # storage and the record model
    "FunctionIdentity",
    "RecordStore",
    "RegressionRecord",
    # comparison
    "ComparisonResult",
    "Diff",
    "ToleranceInfo",
    "compare",
    # extension points
    "Comparator",
    "ComparatorRegistry",
    "Serializer",
    "SerializerRegistry",
    "register_comparator",
    "register_serializer",
    # exceptions and warnings
    "CodeDriftWarning",
    "CorruptRecordError",
    "DeserializationError",
    "IdentityResolutionError",
    "RegressionRecordsError",
    "SerializationError",
    "__version__",
}


def test_public_api_is_exactly_what_is_documented():
    assert set(rr.__all__) == EXPECTED_API


def test_no_name_is_exported_twice():
    assert len(rr.__all__) == len(set(rr.__all__))


@pytest.mark.parametrize("name", sorted(EXPECTED_API))
def test_every_exported_name_resolves(name):
    assert getattr(rr, name) is not None


def test_star_import_matches_all():
    namespace = {}
    exec("from regression_records import *", namespace)  # noqa: S102
    assert set(rr.__all__) <= set(namespace)


def test_version_is_the_one_the_records_are_stamped_with():
    from regression_records.records import SchemaInfo

    assert SchemaInfo().library_version == rr.__version__


@pytest.mark.parametrize(
    "module",
    [
        "regression_records.comparison",
        "regression_records.comparison.paths",
        "regression_records.decorators",
        "regression_records.exceptions",
        "regression_records.identity",
        "regression_records.records",
        "regression_records.replay",
        "regression_records.serialization",
        "regression_records.store",
        "regression_records.testing",
    ],
)
def test_every_submodule_imports_on_its_own(module):
    """Catches an import cycle that only the package's own import order happens to hide."""
    assert importlib.import_module(module) is not None


def test_the_package_ships_a_typing_marker():
    from pathlib import Path

    assert (Path(rr.__file__).parent / "py.typed").exists()
