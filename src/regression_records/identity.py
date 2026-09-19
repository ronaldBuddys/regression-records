"""Derives a stable identity for a function or method being recorded."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .serialization.naming import hash_bytes

_WHITESPACE_RE = re.compile(r"\s+")

Kind = Literal["function", "method", "staticmethod", "classmethod"]


def compute_code_hash(func: Callable[..., Any]) -> str:
    """Hash ``func``'s source, normalized to ignore whitespace/comments.

    Falls back to hashing the compiled bytecode if source isn't available (e.g. for
    builtins or dynamically generated functions). This hash is stored on every record
    as informational metadata only - it is never part of a record's identity/matching
    key, so a code change never orphans historical records; it only lets validation
    surface "this record was captured against different source" as a warning.
    """
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        code = getattr(func, "__code__", None)
        if code is None:
            return "unknown"
        return hash_bytes(code.co_code)

    normalized_lines = []
    for line in source.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            normalized_lines.append(_WHITESPACE_RE.sub(" ", stripped))
    normalized = "\n".join(normalized_lines)
    return hash_bytes(normalized.encode())


@dataclass(frozen=True)
class FunctionIdentity:
    """Identifies a function or method across process runs and code revisions.

    Identity is ``(module, qualname)`` - ``__qualname__`` already encodes class
    nesting (e.g. ``Widget.resize``), which avoids collisions between same-named
    methods on differently-named classes. Moving a function to a different module is
    not auto-resolved (fundamentally ambiguous); see ``RecordStore.rename_identity``.
    """

    module: str
    qualname: str
    kind: Kind
    code_hash: str
    signature: str

    @property
    def path_parts(self) -> list[str]:
        """Directory path components: the module's dotted path, then qualname as leaf."""
        return [*self.module.split("."), self.qualname]

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        kind: Kind = "function",
        name: str | None = None,
    ) -> FunctionIdentity:
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
