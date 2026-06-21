"""Pydantic v2 models for cache entries, hits, and stats."""

from __future__ import annotations

from pydantic import BaseModel


class CacheEntry(BaseModel):
    key_hash: str
    embedding: list[float]
    prompt: str
    response: str
    created_at: float
    ttl: float  # seconds; <= 0 means no expiry
    namespace: str = "default"
    hit_count: int = 0
    last_access: float = 0.0

    def is_expired(self, now: float) -> bool:
        return self.ttl > 0 and (now - self.created_at) > self.ttl


class CacheHit(BaseModel):
    matched_key: str
    similarity: float
    source: str  # "exact" | "semantic"
    saved_cost: float = 0.0
    saved_latency_ms: float = 0.0


class CacheStats(BaseModel):
    hits: int = 0
    misses: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    total_saved_cost: float = 0.0
    total_saved_latency_ms: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class NearMiss(BaseModel):
    """A sub-threshold semantic match — logged for threshold calibration."""

    prompt: str
    best_similarity: float
    matched_prompt: str
