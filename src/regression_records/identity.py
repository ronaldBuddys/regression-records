"""Derives a stable identity for a function or method being recorded."""

from __future__ import annotations

import ast
import inspect
import io
import re
import textwrap
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .serialization.naming import hash_bytes

_WHITESPACE_RE = re.compile(r"\s+")

Kind = Literal["function", "method"]


def compute_code_hash(func: Callable[..., Any]) -> str:
    """Hash ``func``'s source, normalized to ignore whitespace/comments.

    Falls back to hashing the compiled bytecode if source isn't available (e.g. for
    builtins or dynamically generated functions). This hash is stored on every record
    as informational metadata only - it is never part of a record's identity/matching
    key, so a code change never orphans historical records; it only lets validation
    surface "this record was captured against different source" as a warning.

    Parameters
    ----------
    func : Callable
        The function to hash.

    Returns
    -------
    str
        The first 16 hex characters of a sha256 - of the source with decorator lines,
        comments, blank lines and indentation removed and runs of whitespace collapsed,
        or of ``func.__code__.co_code`` when there is no source. ``"unknown"`` when
        there is neither.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        code = getattr(func, "__code__", None)
        if code is None:
            return "unknown"
        return hash_bytes(code.co_code)

    normalized = _normalize_source(source)
    return hash_bytes(normalized.encode())


def _normalize_source(source: str) -> str:
    """Reduce ``source`` to what the function does: no decorators, comments or layout.

    Decorator lines go because they include this package's own - otherwise changing a
    tolerance or a store path would be reported as the function having changed.
    Comments are located with ``tokenize`` rather than by searching for ``#``, which
    also occurs inside string literals: cutting there hid every change after it.
    """
    lines = textwrap.dedent(source).splitlines()
    try:
        definition = ast.parse("\n".join(lines)).body[0]
        # A decorated definition's lineno is its ``def`` line, not its first decorator.
        if isinstance(definition, ast.FunctionDef | ast.AsyncFunctionDef):
            lines = lines[definition.lineno - 1 :]
        tokens = tokenize.generate_tokens(io.StringIO("\n".join(lines)).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                row, col = token.start
                lines[row - 1] = lines[row - 1][:col]
    except (SyntaxError, tokenize.TokenError, IndexError):
        # getsource() returns whole lines, so a lambda's "source" is often a fragment
        # that does not parse on its own. Hash it as it stands: a rare spurious drift
        # warning is a better failure than no hash.
        pass
    stripped = (line.strip() for line in lines)
    return "\n".join(_WHITESPACE_RE.sub(" ", line) for line in stripped if line)


@dataclass(frozen=True)
class FunctionIdentity:
    """Identifies a function or method across process runs and code revisions.

    Identity is ``(module, qualname)`` - ``__qualname__`` already encodes class
    nesting (e.g. ``Widget.resize``), which avoids collisions between same-named
    methods on differently-named classes. Moving a function to a different module is
    not auto-resolved (fundamentally ambiguous); see ``RecordStore.rename_identity``.

    Attributes
    ----------
    module : str
        The defining module's dotted name (``__module__``).
    qualname : str
        The ``__qualname__``, or the ``name=`` override given to the decorator.
    kind : {"function", "method"}
        Which decorator recorded it. Informational: not part of the matching key.
    code_hash : str
        From :func:`compute_code_hash`. Informational: drives the code-drift warning.
    signature : str
        ``str(inspect.signature(func))``, or ``"(...)"``. Informational.
    """

    module: str
    qualname: str
    kind: Kind
    code_hash: str
    signature: str

    @property
    def path_parts(self) -> list[str]:
        """Directory path components: the module's dotted path, then qualname as leaf.

        Returns
        -------
        list[str]
            ``["pkg", "mod", "Widget.resize"]`` for ``pkg.mod:Widget.resize``. The
            qualname stays one component, dots included.
        """
        return [*self.module.split("."), self.qualname]

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        kind: Kind = "function",
        name: str | None = None,
    ) -> FunctionIdentity:
        """Build the identity of ``func``, looking through any decorator wrappers.

        ``name`` overrides the qualname; a callable whose signature cannot be inspected
        is recorded with the placeholder ``"(...)"``.

        Parameters
        ----------
        func : Callable
            The function or method, possibly wrapped by other decorators that set
            ``__wrapped__``.
        kind : {"function", "method"}, default "function"
            Which decorator is recording it.
        name : str, optional
            Used as the qualname in place of ``__qualname__``. The module is always
            that of the unwrapped function.

        Returns
        -------
        FunctionIdentity
            Describing the innermost wrapped function, not ``func`` itself.

        Raises
        ------
        AttributeError
            If the unwrapped callable has no ``__qualname__`` (a ``functools.partial``,
            for one) and no ``name`` was given.
        """
        target = inspect.unwrap(func)
        try:
            signature = str(inspect.signature(target))
        except (TypeError, ValueError):
            signature = "(...)"
        return cls(
            module=target.__module__,
            qualname=name or target.__qualname__,
            kind=kind,
            code_hash=compute_code_hash(target),
            signature=signature,
        )
