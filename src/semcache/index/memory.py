"""In-memory cosine-similarity index over numpy arrays."""

from __future__ import annotations

import numpy as np

from semcache.models import CacheEntry


class MemoryIndex:
    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def add(self, entry: CacheEntry) -> None:
        self._entries[entry.key_hash] = entry

    def get(self, key_hash: str) -> CacheEntry | None:
        return self._entries.get(key_hash)

    def remove(self, key_hash: str) -> None:
        self._entries.pop(key_hash, None)

    def entries(self, namespace: str | None = None) -> list[CacheEntry]:
        if namespace is None:
            return list(self._entries.values())
        return [e for e in self._entries.values() if e.namespace == namespace]

    def clear(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._entries.clear()
        else:
            for k in [k for k, e in self._entries.items() if e.namespace == namespace]:
                del self._entries[k]

    def query(self, embedding: list[float], namespace: str) -> tuple[CacheEntry, float] | None:
        """Return the most similar entry in `namespace` and its cosine score."""
        cands = self.entries(namespace)
        if not cands:
            return None
        q = np.asarray(embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None
        matrix = np.asarray([e.embedding for e in cands], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        sims = (matrix @ q) / (norms * q_norm + 1e-12)
        best = int(np.argmax(sims))
        return cands[best], float(sims[best])

    def __len__(self) -> int:
        return len(self._entries)
