"""Configurable numeric comparison tolerances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from .paths import is_under


class _Unset:
    """Type of :data:`_UNSET`, the default of a field the caller did not pass."""

    def __repr__(self) -> str:
        return "<unset>"


# Typed ``Any`` so it can stand as the default of a ``float`` or ``bool`` field. A
# sentinel rather than "equals the default", so that ``ToleranceConfig(atol=0.0)`` as an
# override still overrides a parent's ``atol=1e-3``.
_UNSET: Final[Any] = _Unset()

#: What a field the caller did not pass resolves to on a config used on its own.
_DEFAULTS: Final[dict[str, float | bool]] = {
    "rtol": 1e-7,
    "atol": 0.0,
    "equal_nan": True,
    "strict_dtype": True,
}


@dataclass(frozen=True)
class ToleranceConfig:
    """Controls how numeric values are compared.

    ``rtol``/``atol`` follow ``numpy.isclose`` semantics: two values are considered
    close if ``abs(actual - expected) <= atol + rtol * abs(expected)``. They apply once
    a float is involved: two integers - scalars, arrays or dataframe columns - are
    always compared exactly.

    ``path_overrides`` maps a dotted/indexed path prefix (as it appears in
    :class:`~regression_records.comparison.base.Diff.path`, e.g. ``"$.output.price"``)
    to a different :class:`ToleranceConfig` to use for that subtree - the most
    specific (longest) matching prefix wins. An override only changes the fields it
    was given: the rest are inherited from the config in effect above it.

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
        ``"$.output"`` / ``"$.state_before"`` / ``"$.state_after"`` under
        replay), then ``.key`` per mapping entry or attribute, ``[0]`` per sequence
        position and ``['name']`` per dataframe column. It applies to that path and to
        everything nested under it, not to paths that merely share its leading
        characters.

    Notes
    -----
    Frozen, because one config is shared by every case of a replay: change one with
    :func:`dataclasses.replace`. Infinities match only an equal infinity, whatever the
    tolerances.

    A config remembers which of ``rtol``, ``atol``, ``equal_nan`` and ``strict_dtype``
    were passed to it; reading one that was not gives its default as listed above. Used
    as an override, only the fields that were passed apply, and the others keep the
    values in effect at the override's path - the root config's, or those of a shorter
    override that also matches. Passing a field explicitly overrides even when the value
    equals the default: ``ToleranceConfig(atol=0.0)`` under a parent with ``atol=1e-3``
    takes that subtree's ``atol`` back to zero. :func:`dataclasses.replace` passes every
    field, so a config derived with it counts all four as given; build an override with
    the constructor.

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

    An override that sets only ``rtol`` keeps the parent's ``atol``:

    >>> tolerance = ToleranceConfig(
    ...     atol=1e-6,
    ...     path_overrides={"$.output.price": ToleranceConfig(rtol=0.01)},
    ... )
    >>> price = tolerance.for_path("$.output.price")
    >>> price.rtol, price.atol
    (0.01, 1e-06)
    """

    rtol: float = _UNSET
    atol: float = _UNSET
    equal_nan: bool = _UNSET
    strict_dtype: bool = _UNSET
    path_overrides: dict[str, ToleranceConfig] = field(default_factory=dict)
    # Which of the _DEFAULTS fields were passed, so an override applies only those.
    _explicit: frozenset[str] = field(
        init=False, repr=False, compare=False, default=frozenset()
    )

    def __post_init__(self) -> None:
        """Record which fields were passed, and give the others their defaults."""
        explicit = frozenset(
            name for name in _DEFAULTS if getattr(self, name) is not _UNSET
        )
        # Frozen: the generated __init__ itself assigns through object.__setattr__.
        object.__setattr__(self, "_explicit", explicit)
        for name, default in _DEFAULTS.items():
            if name not in explicit:
                object.__setattr__(self, name, default)

    def for_path(self, path: str) -> ToleranceConfig:
        """Return the effective config for ``path``, applying every override that matches.

        Parameters
        ----------
        path : str
            A path in the comparison path grammar, e.g. ``"$.output['price']"``.

        Returns
        -------
        ToleranceConfig
            This config when no ``path_overrides`` key is ``path`` or an ancestor of
            it. Otherwise a new config: this one with each matching override laid over
            it in turn, shortest key first, each changing only the fields it was given
            - so the longest key wins where overrides disagree. Its ``path_overrides``
            are this config's merged with every applied override's own (the override's
            winning), so paths further down still resolve, and its given fields are
            the union of the ones given along the way.
        """
        # Called once per node of every compared value, and almost always with no
        # overrides configured at all - so the common case must not pay for the scan.
        if not self.path_overrides:
            return self

        # One override at a time, because each applied override may bring overrides of
        # its own for deeper paths, which then take precedence over the parent's entry
        # for the same key. Only strictly longer keys are taken next, so the loop ends
        # and a shorter key can never undo a more specific one.
        effective = self
        applied_len = -1
        while True:
            matches = [
                prefix
                for prefix in effective.path_overrides
                if len(prefix) > applied_len and is_under(path, prefix)
            ]
            if not matches:
                return effective
            prefix = min(matches, key=len)
            effective = effective._overlay(effective.path_overrides[prefix])
            applied_len = len(prefix)

    def _overlay(self, override: ToleranceConfig) -> ToleranceConfig:
        """Return this config with ``override``'s given fields and overrides on top."""
        # The result governs from here down, so it has to keep answering for_path(): a
        # dataframe comparator resolves each column against the config it was handed.
        # Without the parent's overrides it forgot them, and "$.output" plus
        # "$.output['price']" applied the first to every column and dropped the second.
        merged = ToleranceConfig(
            **{
                name: getattr(override if name in override._explicit else self, name)
                for name in _DEFAULTS
            },
            path_overrides={**self.path_overrides, **override.path_overrides},
        )
        # The constructor counted all four fields as given; keep what really was.
        object.__setattr__(merged, "_explicit", self._explicit | override._explicit)
        return merged
