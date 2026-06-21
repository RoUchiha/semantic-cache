"""Gate 7: Redis backend passes the same index contract. Skipped if unreachable."""

from __future__ import annotations

import pytest

from semcache.models import CacheEntry

redis_index = pytest.importorskip("semcache.index.redis")


@pytest.fixture
def idx():
    try:
        index = redis_index.RedisIndex()
    except Exception as e:  # redis not installed or not running
        pytest.skip(f"Redis unavailable: {e}")
    index.clear()
    yield index
    index.clear()


def _entry(key, emb, ns="default"):
    return CacheEntry(key_hash=key, embedding=emb, prompt=key, response="r",
                      created_at=0, ttl=0, namespace=ns)


def test_redis_contract(idx):
    idx.add(_entry("a", [1.0, 0.0]))
    idx.add(_entry("b", [0.0, 1.0]))
    assert idx.get("a").key_hash == "a"
    best, score = idx.query([1.0, 0.05], "default")
    assert best.key_hash == "a" and score > 0.99
    idx.remove("a")
    assert idx.get("a") is None
    idx.clear()
    assert len(idx) == 0
