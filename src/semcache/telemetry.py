"""Telemetry reporting over a cache's accumulated stats."""

from __future__ import annotations

from semcache.cache import SemanticCache


def report(cache: SemanticCache) -> dict:
    s = cache.stats
    return {
        "lookups": s.lookups,
        "hits": s.hits,
        "misses": s.misses,
        "exact_hits": s.exact_hits,
        "semantic_hits": s.semantic_hits,
        "hit_rate": round(s.hit_rate, 4),
        "saved_cost": round(s.total_saved_cost, 4),
        "saved_latency_ms": round(s.total_saved_latency_ms, 1),
        "near_misses": len(cache.near_misses),
        "size": len(cache.index),
    }
