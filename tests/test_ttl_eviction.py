"""Gate 5: TTL expiry (mocked clock), LRU eviction, namespace isolation."""

from __future__ import annotations

import pytest

from semcache.cache import SemanticCache
from semcache.config import Config
from semcache.embed import MockEmbedder
from tests.conftest import Clock

# Mutually orthogonal vectors -> no accidental semantic collisions.
ORTHO = MockEmbedder(vectors={"p1": [1, 0, 0], "p2": [0, 1, 0], "p3": [0, 0, 1]})


async def _call(_p):
    return "R"


@pytest.mark.asyncio
async def test_ttl_expiry_not_served(embedder):
    clock = Clock(0.0)
    cache = SemanticCache(Config(ttl_seconds=10), embedder=embedder, time_fn=clock)
    calls = {"n": 0}

    async def fn(_p):
        calls["n"] += 1
        return "R"

    await cache.get_or_call("how do i reset my password", fn)  # miss, stored at t=0
    clock.t = 5
    _, hit = await cache.get_or_call("how do i reset my password", fn)
    assert hit is not None and calls["n"] == 1  # still fresh -> hit
    clock.t = 20
    _, hit = await cache.get_or_call("how do i reset my password", fn)
    assert hit is None and calls["n"] == 2  # expired -> miss


@pytest.mark.asyncio
async def test_lru_eviction_respects_max_entries():
    clock = Clock(0.0)
    cache = SemanticCache(Config(max_entries=2, eviction="lru"), embedder=ORTHO, time_fn=clock)

    clock.t = 1
    await cache.get_or_call("p1", _call)
    clock.t = 2
    await cache.get_or_call("p2", _call)
    clock.t = 3
    await cache.get_or_call("p1", _call)  # touch p1 -> newer than p2
    clock.t = 4
    await cache.get_or_call("p3", _call)  # over cap -> evict LRU (p2)

    assert len(cache.index) == 2
    keys = {e.prompt for e in cache.index.entries()}
    assert keys == {"p1", "p3"}  # p2 evicted


@pytest.mark.asyncio
async def test_namespace_isolation(embedder):
    cache = SemanticCache(Config(), embedder=embedder)
    calls = {"n": 0}

    async def fn(_p):
        calls["n"] += 1
        return f"R{calls['n']}"

    await cache.get_or_call("how do i reset my password", fn, namespace="a")
    # Same prompt, different namespace -> must NOT hit a's entry.
    _, hit = await cache.get_or_call("how do i reset my password", fn, namespace="b")
    assert hit is None and calls["n"] == 2
