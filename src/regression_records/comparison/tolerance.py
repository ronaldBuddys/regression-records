"""Configurable numeric comparison tolerances."""

from __future__ import annotations

from dataclasses import dataclass, field

from .paths import is_under


@dataclass
class ToleranceConfig:
    """Controls how numeric values are compared.

    ``rtol``/``atol`` follow ``numpy.isclose`` semantics: two values are considered
    close if ``abs(actual - expected) <= atol + rtol * abs(expected)``.

    ``path_overrides`` maps a dotted/indexed path prefix (as it appears in
    :class:`~regression_records.comparison.base.Diff.path`, e.g. ``"$.output.price"``)
    to a different :class:`ToleranceConfig` to use for that subtree - the most
    specific (longest) matching prefix wins.
    """

    rtol: float = 1e-7
    atol: float = 0.0
    equal_nan: bool = True
    strict_dtype: bool = True
    path_overrides: dict[str, ToleranceConfig] = field(default_factory=dict)

    def for_path(self, path: str) -> ToleranceConfig:
        """Return the effective config for ``path``, applying the most specific override."""
        # Called once per node of every compared value, and almost always with no
        # overrides configured at all - so the common case must not pay for the scan.
        if not self.path_overrides:
            return self

        best: ToleranceConfig | None = None
        best_len = -1
        for prefix, cfg in self.path_overrides.items():
            if is_under(path, prefix) and len(prefix) > best_len:
                best = cfg
                best_len = len(prefix)
        return best if best is not None else self
