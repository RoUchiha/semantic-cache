# semcache — Semantic Cache for LLM Calls

**⚡ [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/rosingh/semantic-cache)** — ask a question two different ways and watch the second serve from cache as a semantic hit.

Returns a stored response when an incoming prompt is **semantically equivalent**
(not just byte-identical) to a previous one — eliminating redundant LLM calls.
Exact-hash fast path, semantic nearest-neighbor lookup, TTL + LRU eviction,
per-namespace isolation, and hit-rate / savings telemetry.

## How it works

1. **Normalize** the prompt (trim, collapse whitespace, optional lowercase).
2. **Exact fast path** — O(1) hash lookup; an exact hit returns *without* embedding.
3. **Semantic lookup** — embed, find nearest neighbor; if cosine ≥ threshold, **hit**.
4. **Miss** — call the LLM, store `(embedding, prompt, response, ttl)`.
5. **Evict** — TTL expiry on read + LRU when over `max_entries`.
6. **Telemetry** — hits/misses, hit-rate, estimated cost + latency saved.

## Correctness guards (a false hit is worse than a miss)

- **Conservative threshold** (default `0.95`) so paraphrase-but-different-intent
  prompts don't collide. Tune it from the **near-miss log** (sub-threshold matches).
- **Per-namespace caches** so different system prompts/tools never cross-contaminate.
- **TTL staleness guard** — expired entries are never served.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -e ".[dev]"

# Benchmark hit-rate + savings on a prompt stream with duplicates/paraphrases
semcache bench --file prompts.jsonl --threshold 0.92
```

```python
from semcache.cache import SemanticCache
from semcache.decorator import cached

cache = SemanticCache()  # HashingEmbedder by default (offline)

@cached(cache, namespace="support")
async def ask_llm(prompt: str) -> str:
    ...  # your real model call
```

Tests use a deterministic embedder (no API key, no model download). The live demo
uses real sentence-transformer embeddings (`pip install -e ".[embeddings]"`).

## Tests

```bash
pytest                 # all gates, offline
pytest --cov=semcache
```

The Redis backend tests are **skipped** if no Redis is reachable (documented).

## Live demo

`app.py` is a Gradio demo using real semantic embeddings: type variations of a
question and watch exact vs semantic hits, similarity scores, and cumulative
savings. Deployable to Hugging Face Spaces.

## Layout

```
src/semcache/
  config.py        # threshold, ttl, max_entries, eviction, backend, savings params
  models.py        # CacheEntry, CacheHit, CacheStats, NearMiss
  normalizer.py    # canonicalization + stable hashing
  embed.py         # Embedder protocol: hashing (default), mock, sentence-transformers
  index/           # CacheIndex protocol: memory (numpy cosine), redis (optional)
  cache.py         # core: exact -> semantic -> miss/store, TTL, LRU, near-miss
  decorator.py     # @cached wrapper
  telemetry.py     # savings report
  cli.py           # bench / stats / clear
```

## Honest about false hits

Semantic caching trades a small false-hit risk for big savings. The default
threshold is deliberately high; the near-miss log exists so you calibrate it on
*your* traffic before lowering it. See [DECISIONS.md](DECISIONS.md).
