"""Environment variables a recorded call reads: captured with it, and set again on replay.

A function that reads ``os.environ`` during a call has an input that is not among its
arguments. Unrecorded, replay runs it under whatever the replaying process has set: a
change of setting reports ``differs`` though the code is fine, and a regression that
shows only under the recorded setting passes. The decorators' ``env=`` names variables
to capture; each record holds their values, and replay sets them around the call.

``secret_env=`` names variables whose values must not be written down. Each is recorded
as a fingerprint, which replay compares with the current value's: it cannot set the
recorded value, only say that it has changed.

A value in ``env=`` is committed in plain text, so two checks stand between a secret
and the store: a name that reads like a secret's is refused when decorating, and a
value shaped like a credential is refused when recording. :class:`plain` vouches for a
variable that trips either by accident.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from ._switch import ENV_VAR
from .exceptions import SecretInEnvError

#: PBKDF2 rounds for a secret's fingerprint: slow enough that guessing a weak secret
#: from its committed fingerprint is expensive, and computed once per value a process
#: sees. Stored in each fingerprint, so changing it leaves old records comparable.
_ROUNDS = 600_000
_SCHEME = "pbkdf2_sha256"

#: A name containing one of these anywhere reads like a secret's.
_SECRET_WORDS = ("SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "APIKEY")
#: These count only as a whole part of the name, between underscores: ``API_KEY`` and
#: ``GITHUB_TOKEN`` are secrets, ``MAX_TOKENS`` and ``KEYBOARD_LAYOUT`` are not.
_SECRET_PARTS = frozenset({"KEY", "TOKEN", "AUTH", "PWD", "DSN"})

#: What a credential looks like, by issuer. Formats specific enough that a setting
#: matching one by chance is unlikely; there is no entropy test, which every hash,
#: id and checksum would fail.
_CREDENTIAL_SHAPES = (
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # scheme://user:password@host - a database or broker URL carrying its password.
    ("a URL holding a password", re.compile(r"[a-zA-Z][\w+.-]*://[^/\s:@]+:[^/\s@]+@")),
    ("an API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    (
        "a GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})"),
    ),
    ("an AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    # Three base64url parts, the first two JSON objects: a signed web token.
    ("a JSON web token", re.compile(r"\beyJ[\w-]{10,}\.eyJ[\w-]{10,}\.[\w-]+")),
)


class plain(str):  # noqa: N801 - reads as a marker in a decorator's arguments
    """Mark a variable in ``env=`` as safe to store in plain text.

    ``env=`` refuses a name that reads like a secret's, and recording refuses a value
    shaped like a credential, because both end up in files you commit. Wrap a name to
    vouch for a variable that trips either check without holding a secret.

    Examples
    --------
    >>> from regression_records import plain
    >>> env = ("PRICE_DECIMALS", plain("SORT_KEY"))
    """

    __slots__ = ()


def env_names(
    names: Sequence[str], argument: str, *, refuse_secrets: bool = False
) -> tuple[str, ...]:
    """Validate the names given to ``env=`` or ``secret_env=``, or held by a record.

    Parameters
    ----------
    names : Sequence[str]
        The variable names.
    argument : str
        ``"env"`` or ``"secret_env"``, for the error message.
    refuse_secrets : bool, default False
        Also refuse a name that reads like a secret's, unless it is wrapped in
        :class:`plain`. For a decorator's ``env=``, whose values are stored in full;
        not for names read back from a record, which must keep loading.

    Returns
    -------
    tuple[str, ...]
        The names, in order, without repeats. One given as :class:`plain` stays so.

    Raises
    ------
    TypeError
        If ``names`` is a single string, which would otherwise be taken letter by
        letter, or holds something other than a string.
    ValueError
        If a name is empty, contains ``=``, or is ``REGRESSION_RECORDS``: replay
        controls the switch itself, and must not be told to set it. With
        ``refuse_secrets``, if a name reads like a secret's.
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
        if (
            refuse_secrets
            and not isinstance(name, plain)
            and _reads_like_a_secret(name)
        ):
            raise ValueError(
                f"{name} reads like a secret, and {argument} stores values in plain "
                f"text, in files you commit. List it in secret_env, which stores a "
                f"fingerprint; or, if its value is safe to commit, write "
                f"plain({str(name)!r})"
            )
    return tuple(dict.fromkeys(names))


def _reads_like_a_secret(name: str) -> bool:
    upper = name.upper()
    if any(word in upper for word in _SECRET_WORDS):
        return True
    return not _SECRET_PARTS.isdisjoint(re.split(r"[^A-Z0-9]+", upper))


def snapshot(names: Sequence[str]) -> dict[str, str | None]:
    """Return each named variable's current value, None for one that is unset.

    Parameters
    ----------
    names : Sequence[str]
        The variables to read.

    Returns
    -------
    dict[str, str | None]
        Keyed by name, in the order given. A name given as :class:`plain` is keyed
        by the bare string, so a record never holds the marker.
    """
    return {str(name): os.environ.get(name) for name in names}


def require_no_credentials(
    names: Sequence[str], values: Mapping[str, str | None]
) -> None:
    """Raise if a variable about to be recorded holds what looks like a credential.

    The last check before a value is written in plain text. It knows a handful of
    credential formats, not every secret: a password with no recognisable shape gets
    through, which is what the name check and ``secret_env=`` are for.

    Parameters
    ----------
    names : Sequence[str]
        The decorator's ``env=`` names. One wrapped in :class:`plain` is not checked.
    values : Mapping[str, str | None]
        Their values, from :func:`snapshot`.

    Raises
    ------
    SecretInEnvError
        Naming the variable and what its value looks like, never the value.
    """
    for name in names:
        value = values[name]
        if value is None or isinstance(name, plain):
            continue
        for looks_like, shape in _CREDENTIAL_SHAPES:
            if shape.search(value):
                raise SecretInEnvError(
                    f"{name} holds what looks like {looks_like}, and env stores "
                    f"values in plain text, in files you commit; nothing was "
                    f"recorded. List it in secret_env, which stores a fingerprint; "
                    f"or, if the value is safe to commit, write plain({str(name)!r})"
                )


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
