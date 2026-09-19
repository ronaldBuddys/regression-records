"""The path grammar naming a location inside a compared value.

Paths are user-facing: every :class:`~regression_records.comparison.base.Diff` carries
one, and :attr:`ToleranceConfig.path_overrides` keys are matched against them. Building
them here rather than by string concatenation at each site keeps the grammar one thing
that can be changed, instead of a convention several modules have to agree on by hand.
"""

from __future__ import annotations

from typing import Any

#: The path assigned to the top-level value being compared.
ROOT_PATH = "$"

#: Roots for the parts of a record compared separately, so one record's diffs stay
#: distinguishable and each part can be targeted by its own tolerance override.
OUTPUT_ROOT = f"{ROOT_PATH}.output"
STATE_BEFORE_ROOT = f"{ROOT_PATH}.state_before"
STATE_AFTER_ROOT = f"{ROOT_PATH}.state_after"


def child(path: str, key: Any) -> str:
    """Extend ``path`` to a mapping entry or object attribute: ``$.price``."""
    return f"{path}.{key}"


def index(path: str, position: int) -> str:
    """Extend ``path`` to a sequence element: ``$[0]``."""
    return f"{path}[{position}]"


def column(path: str, name: Any) -> str:
    """Extend ``path`` to a dataframe column: ``$['price']``."""
    return f"{path}[{name!r}]"


def is_under(path: str, prefix: str) -> bool:
    """Whether ``path`` is ``prefix`` itself or nested somewhere inside it."""
    return (
        path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
    )
