# DECISIONS

Assumptions and deviations made during autonomous execution, dated.

## 2026-06-20

- **Default embedder is HashingEmbedder** (deterministic bag-of-words), so the
  package, tests, and CLI run with zero heavy deps. Real semantic matching uses
  `SentenceTransformerEmbedder` (the `embeddings`/`demo` extra) — that's what the
  live demo uses. Hashing catches lexical/paraphrase overlap; transformers catch
  true synonymy.
- **Conservative threshold (0.95).** A false hit serves a *wrong* answer, which is
  worse than a miss, so the default errs high. The near-miss log records matches in
  `[threshold - margin, threshold)` to calibrate downward safely.
- **Exact path skips embedding** — asserted by a test (the embedder is a spy). This
  is a real latency/cost invariant, not just an internal detail.
- **TTL is checked on read** (lazy expiry) and expired entries are removed when
  encountered; no background sweeper. A `time_fn` is injectable so TTL is tested
  deterministically with a fake clock.
- **Namespaces isolate caches** by being part of the hash key and the query filter,
  so different system prompts/tools never serve each other's responses.
- **Redis backend does client-side cosine** (load namespace entries, compute in
  numpy) rather than requiring a Redis vector module — keeps the dep optional and
  the contract identical to the memory index. Redis tests skip if unreachable.
- **Savings are estimates** from configured `cost_per_call` / `latency_per_call_ms`
  times the hit count — honest accounting, not billing-exact.
