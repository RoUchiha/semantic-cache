"""Gates 2-4: exact fast path (no embed), semantic hit/miss, call-count invariants."""

from __future__ import annotations

import pytest

from semcache.cache import SemanticCache
from semcache.config import Config


@pytest.fixture
def cache(embedder):
    return SemanticCache(Config(threshold=0.95), embedder=embedder)


@pytest.mark.asyncio
async def test_miss_then_exact_hit_skips_embedding(cache, embedder):
    calls = {"n": 0}

    async def fn(p):
        calls["n"] += 1
        return "RESP"

    r1, hit1 = await cache.get_or_call("how do i reset my password", fn)
    assert r1 == "RESP" and hit1 is None and calls["n"] == 1
    embeds_after_miss = len(embedder.calls)

    r2, hit2 = await cache.get_or_call("how do i reset my password", fn)
    assert r2 == "RESP" and hit2.source == "exact" and calls["n"] == 1  # no new call
    assert len(embedder.calls) == embeds_after_miss  # exact path did NOT embed


@pytest.mark.asyncio
async def test_semantic_hit_above_threshold(cache):
    calls = {"n": 0}

    async def fn(p):
        calls["n"] += 1
        return "RESP"

    await cache.get_or_call("how do i reset my password", fn)
    # Paraphrase (cosine ~0.995) -> semantic hit, no new call.
    resp, hit = await cache.get_or_call("how can i reset my password", fn)
    assert resp == "RESP" and hit.source == "semantic" and hit.similarity >= 0.95
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_below_threshold_is_miss(cache):
    calls = {"n": 0}

    async def fn(p):
        calls["n"] += 1
        return f"RESP{calls['n']}"

    await cache.get_or_call("how do i reset my password", fn)
    resp, hit = await cache.get_or_call("what is the capital of france", fn)
    assert hit is None and calls["n"] == 2  # unrelated -> miss
