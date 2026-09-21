"""regression-records: record and regression-test Python function/method executions.

Decorate a function with :func:`record_func` (or a method with :func:`record_method`)
and point it at a directory. The first time each distinct set of arguments is seen, the
call is recorded; every later run of that same call is compared against the recording,
and a difference is reported as a :class:`RegressionMismatchError` with the exact path
that changed::

    from regression_records import record_func

    @record_func("tests/regression_data")
    def score(frame):
        ...

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
from .config import OnMismatch, OnNewCase, OverwritePolicy, RunMode
from .decorators import record_func, record_method
from .exceptions import (
    CodeDriftWarning,
    CorruptRecordError,
    DeserializationError,
    IdentityResolutionError,
    RecordNotFoundError,
    RegressionMismatchError,
    RegressionRecordsError,
    SerializationError,
    UnmatchedInvocationError,
)
from .identity import FunctionIdentity
from .records import RegressionRecord
from .replay import ReplayCase, ReplayReport, ReplayStatus, replay
from .serialization import Serializer, SerializerRegistry, register_serializer
from .store import RecordStore

__all__ = [
    # decorators
    "record_func",
    "record_method",
    # replay
    "replay",
    "ReplayCase",
    "ReplayReport",
    "ReplayStatus",
    # configuration
    "OnMismatch",
    "OnNewCase",
    "OverwritePolicy",
    "RunMode",
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
    "RecordNotFoundError",
    "RegressionMismatchError",
    "RegressionRecordsError",
    "SerializationError",
    "UnmatchedInvocationError",
    "__version__",
]
