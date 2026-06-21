"""Gate 6: @cached decorator, telemetry, near-miss logging."""

from __future__ import annotations

import pytest

from semcache.cache import SemanticCache
from semcache.config import Config
from semcache.decorator import cached
from semcache.telemetry import report


@pytest.mark.asyncio
async def test_decorator_caches(embedder):
    cache = SemanticCache(Config(), embedder=embedder)
    calls = {"n": 0}

    @cached(cache, namespace="support")
    async def ask(prompt: str) -> str:
        calls["n"] += 1
        return "ANSWER"

    assert await ask("how do i reset my password") == "ANSWER"
    assert await ask("how can i reset my password") == "ANSWER"  # paraphrase -> hit
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_telemetry_report(embedder):
    cache = SemanticCache(Config(cost_per_call=0.02), embedder=embedder)

    async def fn(_p):
        return "R"

    await cache.get_or_call("how do i reset my password", fn)   # miss
    await cache.get_or_call("how do i reset my password", fn)   # exact hit
    await cache.get_or_call("how can i reset my password", fn)  # semantic hit
    rep = report(cache)
    assert rep["hits"] == 2 and rep["misses"] == 1
    assert rep["exact_hits"] == 1 and rep["semantic_hits"] == 1
    assert rep["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert rep["saved_cost"] == pytest.approx(0.04)


@pytest.mark.asyncio
async def test_near_miss_logged(embedder):
    # threshold 0.95, margin 0.05 -> near-miss band [0.90, 0.95)
    cache = SemanticCache(Config(threshold=0.95, near_miss_margin=0.05), embedder=embedder)

    async def fn(_p):
        return "R"

    await cache.get_or_call("how do i reset my password", fn)
    # "i want to change my password" has cosine 0.92 with the stored prompt.
    _, hit = await cache.get_or_call("i want to change my password", fn)
    assert hit is None  # below threshold -> miss
    assert len(cache.near_misses) == 1
    assert 0.90 <= cache.near_misses[0].best_similarity < 0.95
