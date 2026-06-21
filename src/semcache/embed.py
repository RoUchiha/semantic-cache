"""Embedding abstraction.

- HashingEmbedder: deterministic bag-of-words hashing vector (no deps) — used in
  tests and as a lightweight fallback.
- MockEmbedder: scripted vectors for precise threshold tests.
- SentenceTransformerEmbedder: real semantic embeddings (lazy import) — used by
  the demo. Install with the `embeddings` extra.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class HashingEmbedder:
    """Bag-of-words hashed into `dim` buckets, L2-normalized. Deterministic."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOKEN.findall(text.lower()):
            vec[hash(tok) % self.dim] += 1.0
        return _l2(vec).tolist()


class MockEmbedder:
    """Returns scripted vectors for known texts; falls back to hashing."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, dim: int = 8) -> None:
        self.vectors = vectors or {}
        self._fallback = HashingEmbedder(dim=dim)
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.vectors:
            return _l2(np.array(self.vectors[text], dtype=np.float64)).tolist()
        return self._fallback.embed(text)


class SentenceTransformerEmbedder:  # pragma: no cover - requires heavy optional dep
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float64).tolist()
