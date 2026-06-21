"""Prompt canonicalization + stable hashing for the exact fast path."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize(prompt: str, lower: bool = True) -> str:
    """Canonical form: trim, collapse internal whitespace, optional lowercase.
    Whitespace/case variants of the same prompt normalize to one string."""
    text = _WS.sub(" ", prompt.strip())
    return text.lower() if lower else text


def key_hash(normalized: str, namespace: str = "default") -> str:
    """Stable per-namespace hash of a normalized prompt."""
    return hashlib.sha256(f"{namespace}\x00{normalized}".encode()).hexdigest()
