"""regression-records: record what Python functions do, and replay it later.

Decorate a function with :func:`record_func` (or a method with :func:`record_method`)
and give it a directory, or leave it out and set ``REGRESSION_RECORDS_STORE`` (in the
environment or a ``.env`` file). Nothing is recorded until recording is switched on, with
the environment variable ``REGRESSION_RECORDS=1`` or :func:`recording`; then each call's
arguments and result are written as a record. Later - after a code change, a library
upgrade, on another machine - :func:`replay` reruns every recorded call against the
current code and reports what changed, with the exact path of each difference::

    from regression_records import record_func, replay

    @record_func("tests/regression_data")
    def score(frame):
        ...

    # REGRESSION_RECORDS=1 python run_pipeline.py      records every call
    report = replay("tests/regression_data")
    assert report.passed, report.summary()

Everything re-exported here is the supported API. Names reachable only through the
submodules are internal and may change without notice - with two deliberate exceptions
worth knowing about: the record model in :mod:`regression_records.records`
(:class:`~regression_records.records.Invocation`,
:class:`~regression_records.records.SerializedValue`,
:class:`~regression_records.records.ExceptionRecord`) is stable enough to inspect a
loaded record with, and :mod:`regression_records.comparison.paths` documents the diff
path grammar that :attr:`ToleranceConfig.path_overrides` keys are matched against.
"""

from __future__ import annotations

from ._switch import disable, enable, recording
from ._version import __version__
from .comparison import (
    Comparator,
    ComparatorRegistry,
    ComparisonResult,
    Diff,
    ToleranceConfig,
    ToleranceInfo,
    compare,
    register_comparator,
)
from .decorators import record_func, record_method
from .exceptions import (
    CodeDriftWarning,
    CorruptRecordError,
    DeserializationError,
    IdentityResolutionError,
    RegressionRecordsError,
    SerializationError,
    StoreNotConfiguredError,
)
from .identity import FunctionIdentity
from .manifest import Environment, RecordingSession, environment, session
from .records import RegressionRecord
from .replay import ReplayCase, ReplayReport, ReplayStatus, replay
from .serialization import Serializer, SerializerRegistry, register_serializer
from .store import RecordStore
from .testing import replay_tests

__all__ = [
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
    "StoreNotConfiguredError",
    "__version__",
]
