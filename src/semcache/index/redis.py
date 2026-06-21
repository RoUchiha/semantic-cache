"""Optional Redis-backed index — same contract as MemoryIndex, but durable.

Vector search is done client-side (load namespace entries, cosine in numpy), so
it needs no Redis vector module. Requires the `redis` extra; import is lazy.
"""

from __future__ import annotations

import numpy as np

from semcache.models import CacheEntry

_PREFIX = "semcache"


class RedisIndex:
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        import redis  # lazy

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()  # fail fast if unreachable

    def _key(self, namespace: str, key_hash: str) -> str:
        return f"{_PREFIX}:{namespace}:{key_hash}"

    def add(self, entry: CacheEntry) -> None:
        self._r.set(self._key(entry.namespace, entry.key_hash), entry.model_dump_json())

    def get(self, key_hash: str) -> CacheEntry | None:
        for k in self._r.scan_iter(f"{_PREFIX}:*:{key_hash}"):
            raw = self._r.get(k)
            return CacheEntry.model_validate_json(raw) if raw else None
        return None

    def remove(self, key_hash: str) -> None:
        for k in list(self._r.scan_iter(f"{_PREFIX}:*:{key_hash}")):
            self._r.delete(k)

    def entries(self, namespace: str | None = None) -> list[CacheEntry]:
        pattern = f"{_PREFIX}:{namespace}:*" if namespace else f"{_PREFIX}:*"
        out = []
        for k in self._r.scan_iter(pattern):
            raw = self._r.get(k)
            if raw:
                out.append(CacheEntry.model_validate_json(raw))
        return out

    def clear(self, namespace: str | None = None) -> None:
        pattern = f"{_PREFIX}:{namespace}:*" if namespace else f"{_PREFIX}:*"
        keys = list(self._r.scan_iter(pattern))
        if keys:
            self._r.delete(*keys)

    def query(self, embedding: list[float], namespace: str) -> tuple[CacheEntry, float] | None:
        cands = self.entries(namespace)
        if not cands:
            return None
        q = np.asarray(embedding, dtype=np.float64)
        qn = np.linalg.norm(q)
        if qn == 0:
            return None
        matrix = np.asarray([e.embedding for e in cands], dtype=np.float64)
        sims = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * qn + 1e-12)
        best = int(np.argmax(sims))
        return cands[best], float(sims[best])

    def __len__(self) -> int:
        return sum(1 for _ in self._r.scan_iter(f"{_PREFIX}:*"))
