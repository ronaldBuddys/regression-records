"""Configurable numeric comparison tolerances."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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

    Attributes
    ----------
    rtol : float, default 1e-7
        Relative tolerance, as a fraction of the *expected* (recorded) value's
        magnitude - ``0.01`` allows 1%. Not symmetric: the actual value does not scale
        it.
    atol : float, default 0.0
        Absolute tolerance, in the units of the values compared. With the default of
        zero, an expected value of exactly ``0.0`` only matches ``0.0``.
    equal_nan : bool, default True
        Whether NaN matches NaN. A NaN on one side only is always a difference.
    strict_dtype : bool, default True
        Whether arrays, numpy scalars, Series and dataframe columns must also agree in
        dtype. When False only their values (and a dataframe's column names) are
        compared. Plain Python numbers are unaffected: ``1`` and ``1.0`` compare by
        value either way.
    path_overrides : dict[str, ToleranceConfig], default {}
        Path prefix to the config governing that subtree. A key is a path in the
        grammar of :mod:`regression_records.comparison.paths` - the root (``"$"``, or
        ``"$.output"`` / ``"$.state_before"`` / ``"$.state_after"`` under the
        decorators), then ``.key`` per mapping entry or attribute, ``[0]`` per sequence
        position and ``['name']`` per dataframe column. It applies to that path and to
        everything nested under it, not to paths that merely share its leading
        characters.

    Notes
    -----
    Infinities match only an equal infinity, whatever the tolerances. An override
    replaces the whole config for its subtree: fields it leaves at their defaults are
    the defaults, not the parent's values.

    Examples
    --------
    >>> from regression_records.comparison.tolerance import ToleranceConfig
    >>> tolerance = ToleranceConfig(
    ...     rtol=1e-9,
    ...     path_overrides={"$.output.price": ToleranceConfig(rtol=0.01)},
    ... )
    >>> tolerance.for_path("$.output.price[0]").rtol
    0.01
    >>> tolerance.for_path("$.output.prices").rtol
    1e-09
    """

    rtol: float = 1e-7
    atol: float = 0.0
    equal_nan: bool = True
    strict_dtype: bool = True
    path_overrides: dict[str, ToleranceConfig] = field(default_factory=dict)

    def for_path(self, path: str) -> ToleranceConfig:
        """Return the effective config for ``path``, applying the most specific override.

        Parameters
        ----------
        path : str
            A path in the comparison path grammar, e.g. ``"$.output['price']"``.

        Returns
        -------
        ToleranceConfig
            This config when no ``path_overrides`` key is ``path`` or an ancestor of
            it; otherwise the override with the longest such key. An override with no
            ``path_overrides`` of its own is returned as a copy carrying this config's,
            so that paths further down still resolve.
        """
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
        if best is None:
            return self
        if best.path_overrides:
            return best
        # The override governs from here down, so it has to keep answering for_path():
        # a dataframe comparator resolves each column against the config it was handed.
        # Without the parent's overrides it forgot them, and "$.output" plus
        # "$.output['price']" applied the first to every column and dropped the second.
        return replace(best, path_overrides=self.path_overrides)
