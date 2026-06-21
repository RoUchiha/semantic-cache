"""Gate 1 + 2 (part): config loads; normalizer canonicalizes; stable hashing."""

from __future__ import annotations

from semcache.config import Config
from semcache.normalizer import key_hash, normalize


def test_config_defaults_and_load(tmp_path):
    assert Config().threshold == 0.95
    p = tmp_path / "c.yaml"
    p.write_text("threshold: 0.9\nttl_seconds: 60\nmax_entries: 5\n", encoding="utf-8")
    c = Config.load(p)
    assert c.threshold == 0.9 and c.ttl_seconds == 60 and c.max_entries == 5


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello   World  ") == "hello world"
    assert normalize("Hello World") == normalize("hello   world")


def test_key_hash_stable_and_namespaced():
    a = key_hash(normalize("Hi there"), "ns1")
    b = key_hash(normalize("hi    there"), "ns1")
    c = key_hash(normalize("Hi there"), "ns2")
    assert a == b           # whitespace/case variants collide
    assert a != c           # namespaces don't
