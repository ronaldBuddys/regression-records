"""Environment variables a recorded call reads: captured with it, and set again on replay.

A function that reads ``os.environ`` during a call has an input that is not among its
arguments. Unrecorded, replay runs it under whatever the replaying process has set: a
change of setting reports ``differs`` though the code is fine, and a regression that
shows only under the recorded setting passes. The decorators' ``env=`` names variables
to capture; each record holds their values, and replay sets them around the call.

``secret_env=`` names variables whose values must not be written down. Each is recorded
as a fingerprint, which replay compares with the current value's: it cannot set the
recorded value, only say that it has changed.
"""

from __future__ import annotations

import functools
import hashlib
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from ._switch import ENV_VAR

#: PBKDF2 rounds for a secret's fingerprint: slow enough that guessing a weak secret
#: from its committed fingerprint is expensive, and computed once per value a process
#: sees. Stored in each fingerprint, so changing it leaves old records comparable.
_ROUNDS = 600_000
_SCHEME = "pbkdf2_sha256"


def env_names(names: Sequence[str], argument: str) -> tuple[str, ...]:
    """Validate the names given to ``env=`` or ``secret_env=``.

    Parameters
    ----------
    names : Sequence[str]
        The variable names.
    argument : str
        ``"env"`` or ``"secret_env"``, for the error message.

    Returns
    -------
    tuple[str, ...]
        The names, in order, without repeats.

    Raises
    ------
    TypeError
        If ``names`` is a single string, which would otherwise be taken letter by
        letter, or holds something other than a string.
    ValueError
        If a name is empty, contains ``=``, or is ``REGRESSION_RECORDS``: replay
        controls the switch itself, and must not be told to set it.
    """
    if isinstance(names, str):
        raise TypeError(
            f"{argument} takes a sequence of variable names, not one string; write "
            f"{argument}=({names!r},)"
        )
    names = tuple(names)  # a generator would be used up by the checks below
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"{argument} names must be strings, not {type(name).__name__}"
            )
        if not name or "=" in name:
            raise ValueError(f"{name!r} is not an environment variable name")
        if name == ENV_VAR:
            raise ValueError(
                f"{ENV_VAR} is the recording switch, which replay sets itself; it "
                f"cannot be listed in {argument}"
            )
    return tuple(dict.fromkeys(names))


def snapshot(names: Sequence[str]) -> dict[str, str | None]:
    """Return each named variable's current value, None for one that is unset.

    Parameters
    ----------
    names : Sequence[str]
        The variables to read.

    Returns
    -------
    dict[str, str | None]
        Keyed by name, in the order given.
    """
    return {name: os.environ.get(name) for name in names}


def fingerprint(name: str, value: str, rounds: int = _ROUNDS) -> str:
    """Return a one-way fingerprint of a secret's value, to compare without keeping it.

    PBKDF2-SHA256, salted with the variable's name so that equal values under
    different names do not match, and slow so that a weak secret is costly to guess.
    A long random key is safe to fingerprint this way; a short or guessable password
    can still be found by trying candidates, so do not commit fingerprints of those.

    Parameters
    ----------
    name : str
        The variable's name.
    value : str
        Its value.
    rounds : int, optional
        PBKDF2 rounds; the current default when recording.

    Returns
    -------
    str
        ``"pbkdf2_sha256$<rounds>$<hex digest>"``.
    """
    return _fingerprint(name, value, rounds)


@functools.lru_cache(maxsize=256)
def _fingerprint(name: str, value: str, rounds: int) -> str:
    # Cached: a recorded function is called many times with the same environment, and
    # each fingerprint costs a tenth of a second.
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        value.encode(),
        f"regression-records/secret_env/{name}".encode(),
        rounds,
    )
    return f"{_SCHEME}${rounds}${digest.hex()}"


def changed_secrets(recorded: Mapping[str, str | None]) -> tuple[str, ...]:
    """Return the names whose current value does not match its recorded fingerprint.

    Parameters
    ----------
    recorded : Mapping[str, str | None]
        A record's ``secret_env``: fingerprints, None for a variable that was unset.

    Returns
    -------
    tuple[str, ...]
        The names that changed, in the record's order. Set where it was unset, unset
        where it was set, and a different value all count.
    """
    changed = []
    for name, expected in recorded.items():
        value = os.environ.get(name)
        if expected is None or value is None:
            if (expected is None) != (value is None):
                changed.append(name)
            continue
        scheme, _, rest = expected.partition("$")
        rounds, _, _ = rest.partition("$")
        if scheme != _SCHEME or not rounds.isdigit():
            changed.append(name)  # not a fingerprint this version can check
        elif fingerprint(name, value, int(rounds)) != expected:
            changed.append(name)
    return tuple(changed)


@contextmanager
def applied(recorded: Mapping[str, str | None]) -> Iterator[None]:
    """Set the recorded variables for the duration, then put back what was there.

    A variable recorded as unset is removed. ``os.environ`` is process-wide, so
    another thread reading these variables meanwhile sees the recorded values too.

    Parameters
    ----------
    recorded : Mapping[str, str | None]
        A record's ``env``.
    """
    previous = snapshot(list(recorded))
    try:
        _set(recorded)
        yield
    finally:
        _set(previous)


def _set(values: Mapping[str, str | None]) -> None:
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
