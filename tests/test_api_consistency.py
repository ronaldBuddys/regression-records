"""Small API behaviours: tolerance overrides, registry priority, partial identities."""

import dataclasses
import functools

import pytest

from regression_records.comparison import ToleranceConfig
from regression_records.comparison.base import ComparatorRegistry
from regression_records.identity import FunctionIdentity
from regression_records.serialization.base import SerializerRegistry

# --- ToleranceConfig ------------------------------------------------------------------


def test_tolerance_config_is_frozen():
    config = ToleranceConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.rtol = 0.5
    assert dataclasses.replace(config, rtol=0.5).rtol == 0.5


def test_an_override_with_overrides_of_its_own_keeps_the_parents():
    # A frame at "$.frame" resolves its columns against the config resolved for it.
    root = ToleranceConfig(
        path_overrides={
            "$.frame": ToleranceConfig(
                atol=1.0, path_overrides={"$.frame['b']": ToleranceConfig(atol=2.0)}
            ),
            "$.frame['c']": ToleranceConfig(atol=3.0),
        }
    )

    frame = root.for_path("$.frame")

    assert frame.atol == 1.0
    assert frame.for_path("$.frame['a']").atol == 1.0
    assert frame.for_path("$.frame['b']").atol == 2.0
    assert frame.for_path("$.frame['c']").atol == 3.0  # was dropped: fell back to 1.0


def test_an_overrides_own_entry_wins_over_the_parents_for_the_same_path():
    inner = {"$.x.y": ToleranceConfig(atol=2.0)}
    root = ToleranceConfig(
        path_overrides={
            "$.x": ToleranceConfig(path_overrides=inner),
            "$.x.y": ToleranceConfig(atol=9.0),
        }
    )

    assert root.for_path("$.x").for_path("$.x.y").atol == 2.0


# --- registry priority ----------------------------------------------------------------


class _Entry:
    """Serves as a serializer or comparator; only identity and tag matter here."""

    def __init__(self, tag):
        self.type_tag = tag

    def can_handle(self, *values):
        return True


def _registry_with_a_fallback(kind):
    registry = kind()
    registry.register(_Entry("ordinary"))
    registry.register(_Entry("fallback"), fallback=True)
    return registry


@pytest.mark.parametrize("kind", [SerializerRegistry, ComparatorRegistry])
@pytest.mark.parametrize(
    ("priority", "fallback"),
    [(2, False), (-1, False), (0, True), (3, True)],
    ids=["ordinary-after-fallback", "negative", "fallback-first", "past-the-end"],
)
def test_a_priority_outside_the_entrys_own_segment_is_refused(kind, priority, fallback):
    registry = _registry_with_a_fallback(kind)

    with pytest.raises(ValueError, match="out of range"):
        registry.register(_Entry("new"), priority=priority, fallback=fallback)
    assert len(registry) == 2


def _order(registry):
    return [entry.type_tag for entry in registry._entries]


@pytest.mark.parametrize("kind", [SerializerRegistry, ComparatorRegistry])
def test_priorities_within_the_segment_are_honoured(kind):
    registry = _registry_with_a_fallback(kind)

    registry.register(_Entry("first"), priority=0)
    registry.register(_Entry("last-fallback"), priority=3, fallback=True)
    registry.register(_Entry("default"))  # after the ordinary ones, before fallbacks

    assert _order(registry) == [
        "first",
        "ordinary",
        "default",
        "fallback",
        "last-fallback",
    ]


# --- identity -------------------------------------------------------------------------


def _function(a, b):
    return a + b


def test_a_partial_is_refused():
    bound = functools.partial(_function, 1)

    with pytest.raises(TypeError, match="no __qualname__"):
        FunctionIdentity.from_function(bound)
