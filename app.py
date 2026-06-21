"""Gradio live demo for semcache.

Ask a question, then ask it again differently — watch it serve from cache as a
**semantic** hit (with the similarity score) instead of re-calling the LLM, and
see cumulative savings climb.

Uses real sentence-transformer embeddings when available (installed on the Space),
and falls back to the deterministic hashing embedder otherwise. Deployable to HF
Spaces.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr  # noqa: E402

from semcache.cache import SemanticCache  # noqa: E402
from semcache.config import Config  # noqa: E402
from semcache.telemetry import report  # noqa: E402


def _build_embedder():
    try:
        from semcache.embed import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(), "sentence-transformers (all-MiniLM-L6-v2)"
    except Exception:
        from semcache.embed import HashingEmbedder

        return HashingEmbedder(), "hashing fallback (lexical only)"


EMBEDDER, EMBED_NAME = _build_embedder()
# Real embeddings score paraphrases lower than the strict 0.95 default; 0.80 is a
# sensible demo threshold. (Production default stays conservative at 0.95.)
CACHE = SemanticCache(Config(threshold=0.80, cost_per_call=0.01), embedder=EMBEDDER)
_COUNTER = {"n": 0}


async def _fake_llm(prompt: str) -> str:
    _COUNTER["n"] += 1
    return f"[freshly generated answer #{_COUNTER['n']} for: {prompt}]"


def ask(prompt: str):
    if not prompt.strip():
        return "Enter a question.", _stats_md()
    response, hit = asyncio.run(CACHE.get_or_call(prompt, _fake_llm))
    if hit is None:
        verdict = "### ❌ MISS — called the LLM and cached the result"
    elif hit.source == "exact":
        verdict = "### ⚡ EXACT HIT — served from cache (no embedding, no LLM call)"
    else:
        verdict = (f"### 🟢 SEMANTIC HIT — similarity {hit.similarity:.3f} "
                   f"(no LLM call, saved ${hit.saved_cost:.3f})")
    return f"{verdict}\n\n**Response:** {response}", _stats_md()


def _stats_md() -> str:
    r = report(CACHE)
    return (
        f"**Lookups:** {r['lookups']}  ·  **Hit rate:** {r['hit_rate'] * 100:.0f}%  ·  "
        f"**Exact:** {r['exact_hits']}  ·  **Semantic:** {r['semantic_hits']}  ·  "
        f"**Saved:** ${r['saved_cost']:.2f}  ·  **Entries:** {r['size']}"
    )


with gr.Blocks(title="Semantic Cache", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ⚡ Semantic Cache for LLM Calls\n"
        "Equivalent prompts shouldn't cost two LLM calls. Ask a question, then ask it "
        "**a different way** — the cache serves the stored answer as a *semantic* hit.\n\n"
        f"_Embedder: {EMBED_NAME} · demo threshold 0.80_"
    )
    with gr.Row():
        q = gr.Textbox(label="Ask", placeholder="How do I reset my password?", scale=4)
        btn = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(
        ["How do I reset my password?", "How can I reset my password?",
         "What's the process to change my password?", "What is the capital of France?"],
        inputs=q,
    )
    out = gr.Markdown()
    stats = gr.Markdown()
    btn.click(ask, q, [out, stats])
    q.submit(ask, q, [out, stats])


if __name__ == "__main__":
    demo.launch()
