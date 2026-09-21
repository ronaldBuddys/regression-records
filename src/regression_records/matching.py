"""Deciding which historical record - if any - an invocation corresponds to.

An invocation is keyed by an ``input_hash``: the sha256 of its *serialized* arguments.
Serializing first is what makes the key meaningful, because external values are already
content-addressed by then (two equal numpy arrays serialize to the same ``.npy``
reference regardless of object identity), so equal inputs hash equally across processes
and runs.

Matching is always **exact**. Comparison tolerances apply to a call's *output* and
*state*, never to its inputs: two invocations with slightly different arguments are two
different cases to record, not one case that nearly matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .config import OnNewCase, RunMode
from .exceptions import RecordNotFoundError, UnmatchedInvocationError
from .identity import FunctionIdentity
from .records import RegressionRecord
from .store import RecordStore


def canonical_payload(args_tree: Any, kwargs_tree: Any) -> str:
    """Render a serialized invocation as the canonical string that gets hashed.

    ``sort_keys`` makes dict ordering irrelevant and the compact separators keep the
    encoding stable, so the same inputs always produce the same bytes.

    Parameters
    ----------
    args_tree : Any
        The positional arguments, already converted by ``to_json_tree``.
    kwargs_tree : Any
        The keyword arguments, likewise.

    Returns
    -------
    str
        Compact JSON of ``{"args": ..., "kwargs": ...}`` with sorted keys.

    Raises
    ------
    TypeError
        If a tree holds something ``json`` cannot encode, i.e. it was not serialized
        first.
    """
    return json.dumps(
        {"args": args_tree, "kwargs": kwargs_tree},
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_input_hash(args_tree: Any, kwargs_tree: Any) -> str:
    """Hash a serialized invocation's arguments.

    The identity of the function is deliberately *not* part of the hash - records are
    already namespaced by identity on disk, and mixing it in would needlessly change
    every hash whenever a function moved module.

    Parameters
    ----------
    args_tree : Any
        The positional arguments, already converted by ``to_json_tree``.
    kwargs_tree : Any
        The keyword arguments, likewise.

    Returns
    -------
    str
        The 64-character sha256 hex digest of :func:`canonical_payload`.

    Raises
    ------
    TypeError
        If a tree holds something ``json`` cannot encode, i.e. it was not serialized
        first.
    """
    return hashlib.sha256(
        canonical_payload(args_tree, kwargs_tree).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Match:
    """The outcome of looking an invocation up in the store.

    Attributes
    ----------
    input_hash : str
        The full sha256 hex digest the invocation was looked up by.
    record : RegressionRecord or None
        The record of these exact inputs, or None if there is none.
    """

    input_hash: str
    record: RegressionRecord | None
    #: Whether *any* record exists for this identity, which distinguishes "this function
    #: has never been recorded" from "this function has been recorded, but not with
    #: these inputs".
    identity_has_records: bool

    @property
    def is_new_case(self) -> bool:
        """Return whether no record exists for these inputs.

        Returns
        -------
        bool
            True when ``record`` is None.
        """
        return self.record is None


def find_match(
    store: RecordStore, identity: FunctionIdentity, args_tree: Any, kwargs_tree: Any
) -> Match:
    """Look up the historical record for an already-serialized invocation.

    Parameters
    ----------
    store : RecordStore
        The store to search.
    identity : FunctionIdentity
        The function or method that was called.
    args_tree : Any
        The positional arguments, already converted by ``to_json_tree``.
    kwargs_tree : Any
        The keyword arguments, likewise.

    Returns
    -------
    Match
        The input hash, the record found (or None), and whether ``identity`` has any
        record at all - only looked up separately when none matched.

    Raises
    ------
    CorruptRecordError
        If a candidate record file cannot be read or is malformed.
    ValueError
        If ``identity`` cannot be used as a store path.
    """
    input_hash = compute_input_hash(args_tree, kwargs_tree)
    record = store.find_by_input_hash(identity, input_hash)
    return Match(
        input_hash=input_hash,
        record=record,
        identity_has_records=True
        if record is not None
        else store.has_records(identity),
    )


class Action(StrEnum):
    """What a decorated call should do once its match is known."""

    #: Write a record for this invocation.
    RECORD = "record"
    #: Compare this invocation's result against ``match.record``.
    VALIDATE = "validate"
    #: Call through without recording or validating.
    SKIP = "skip"


def decide_action(match: Match, *, mode: RunMode, on_new_case: OnNewCase) -> Action:
    """Resolve ``mode`` and ``on_new_case`` against a lookup result.

    ``on_new_case`` must already be resolved (see
    :func:`~regression_records.config.resolve_on_new_case`), which is what lets this
    read as three independent questions rather than a grid of mode/policy combinations:
    is recording all we were asked for, did we find a match, and if not, what did the
    caller want done about it.

    Raises :class:`RecordNotFoundError` when validating a function that has never been
    recorded, and :class:`UnmatchedInvocationError` when a new case is not permitted.
    Both carry the diagnostic the caller needs, so they are raised here rather than
    returned as another action for the engine to re-raise identically.

    Whether a *matched* record may be replaced on disk is a separate question, decided
    by :class:`~regression_records.config.OverwritePolicy` once the comparison result is
    known.

    Parameters
    ----------
    match : Match
        The lookup result from :func:`find_match`.
    mode : RunMode
        The decoration's run mode, as an enum member - this function dispatches on
        identity, so a plain string matches nothing.
    on_new_case : OnNewCase
        The resolved new-case policy; never None.

    Returns
    -------
    Action
        ``RECORD`` under ``RunMode.RECORD_ONLY``, matched or not. Otherwise
        ``VALIDATE`` for a matched invocation, and for a new case ``RECORD`` or
        ``SKIP`` as ``on_new_case`` says.

    Raises
    ------
    RecordNotFoundError
        Under ``RunMode.VALIDATE_ONLY``, if the identity has no records at all -
        whatever ``on_new_case`` says.
    UnmatchedInvocationError
        If the invocation is a new case and ``on_new_case`` is ``OnNewCase.ERROR``.
    """
    if mode is RunMode.RECORD_ONLY:
        return Action.RECORD

    if not match.is_new_case:
        return Action.VALIDATE

    if mode is RunMode.VALIDATE_ONLY and not match.identity_has_records:
        raise RecordNotFoundError(
            "no records exist for this function, so there is nothing to validate "
            "against; record a run first (mode=RunMode.RECORD_ONLY or RunMode.BOTH) "
            "before validating"
        )

    if on_new_case is OnNewCase.IGNORE:
        return Action.SKIP
    if on_new_case is OnNewCase.RECORD:
        # resolve_on_new_case() has already rejected RECORD under VALIDATE_ONLY, so
        # reaching here means writing a new case really is permitted.
        return Action.RECORD
    raise UnmatchedInvocationError(
        f"no record matches these inputs (input_hash={match.input_hash[:16]}) and "
        f"on_new_case=OnNewCase.ERROR"
        + (
            " (the default under mode=RunMode.VALIDATE_ONLY); use RunMode.BOTH to "
            "record new cases as they appear, or OnNewCase.IGNORE to pass them through"
            if mode is RunMode.VALIDATE_ONLY
            else ""
        )
    )
