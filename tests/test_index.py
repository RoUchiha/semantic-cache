"""Gate 3 (part): memory index nearest-neighbor ordering."""

from __future__ import annotations

from semcache.index.memory import MemoryIndex
from semcache.models import CacheEntry


def _entry(key: str, emb: list[float], ns: str = "default") -> CacheEntry:
    return CacheEntry(key_hash=key, embedding=emb, prompt=key, response="r",
                      created_at=0, ttl=0, namespace=ns)


def test_query_returns_nearest():
    idx = MemoryIndex()
    idx.add(_entry("a", [1.0, 0.0]))
    idx.add(_entry("b", [0.0, 1.0]))
    idx.add(_entry("c", [0.9, 0.1]))
    best, score = idx.query([1.0, 0.05], "default")
    assert best.key_hash == "a"
    assert score > 0.99


def test_query_namespace_isolation():
    idx = MemoryIndex()
    idx.add(_entry("a", [1.0, 0.0], ns="ns1"))
    idx.add(_entry("b", [1.0, 0.0], ns="ns2"))
    best, _ = idx.query([1.0, 0.0], "ns2")
    assert best.key_hash == "b"


def test_query_empty_namespace():
    assert MemoryIndex().query([1.0, 0.0], "default") is None
