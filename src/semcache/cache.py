"""SemanticCache core: exact fast path -> semantic NN -> miss/store.

Correctness guards: per-namespace isolation, TTL expiry on read, conservative
similarity threshold, and a near-miss log for calibration.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable

from loguru import logger

from semcache.config import Config
from semcache.embed import Embedder
from semcache.index.memory import MemoryIndex
from semcache.models import CacheEntry, CacheHit, CacheStats, NearMiss
from semcache.normalizer import key_hash, normalize

CallFn = Callable[[str], str] | Callable[[str], Awaitable[str]]


class SemanticCache:
    def __init__(
        self,
        config: Config | None = None,
        embedder: Embedder | None = None,
        index: MemoryIndex | None = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        from semcache.embed import HashingEmbedder

        self.config = config or Config()
        self.embedder = embedder or HashingEmbedder()
        self.index = index or MemoryIndex()
        self.time_fn = time_fn
        self.stats = CacheStats()
        self.near_misses: list[NearMiss] = []

    async def get_or_call(
        self, prompt: str, call_fn: CallFn, namespace: str = "default"
    ) -> tuple[str, CacheHit | None]:
        normalized = normalize(prompt, self.config.normalize_case)
        kh = key_hash(normalized, namespace)
        now = self.time_fn()

        # 1. Exact fast path — no embedding call.
        entry = self.index.get(kh)
        if entry is not None:
            if entry.is_expired(now):
                self.index.remove(kh)
            else:
                return entry.response, self._hit(entry, similarity=1.0, source="exact", now=now)

        # 2. Semantic lookup.
        embedding = self.embedder.embed(normalized)
        match = self.index.query(embedding, namespace)
        if match is not None:
            best, score = match
            if best.is_expired(now):
                self.index.remove(best.key_hash)
            elif score >= self.config.threshold:
                return best.response, self._hit(best, similarity=score, source="semantic", now=now)
            elif score >= self.config.threshold - self.config.near_miss_margin:
                self.near_misses.append(
                    NearMiss(prompt=prompt, best_similarity=score, matched_prompt=best.prompt)
                )

        # 3. Miss — call the underlying function and store.
        result = call_fn(prompt)
        response = await result if inspect.isawaitable(result) else result

        self.index.add(
            CacheEntry(
                key_hash=kh, embedding=embedding, prompt=prompt, response=response,
                created_at=now, ttl=self.config.ttl_seconds, namespace=namespace,
                hit_count=0, last_access=now,
            )
        )
        self._evict_if_needed()
        self.stats.misses += 1
        logger.debug("cache miss for namespace={} (size={})", namespace, len(self.index))
        return response, None

    def _hit(self, entry: CacheEntry, similarity: float, source: str, now: float) -> CacheHit:
        entry.hit_count += 1
        entry.last_access = now
        self.stats.hits += 1
        if source == "exact":
            self.stats.exact_hits += 1
        else:
            self.stats.semantic_hits += 1
        self.stats.total_saved_cost += self.config.cost_per_call
        self.stats.total_saved_latency_ms += self.config.latency_per_call_ms
        return CacheHit(
            matched_key=entry.key_hash, similarity=similarity, source=source,
            saved_cost=self.config.cost_per_call, saved_latency_ms=self.config.latency_per_call_ms,
        )

    def _evict_if_needed(self) -> None:
        if self.config.eviction != "lru":
            return
        while len(self.index) > self.config.max_entries:
            oldest = min(self.index.entries(), key=lambda e: e.last_access)
            self.index.remove(oldest.key_hash)
            logger.debug("evicted LRU entry {}", oldest.key_hash[:8])

    def clear(self, namespace: str | None = None) -> None:
        self.index.clear(namespace)
