"""Typer CLI: bench / stats / clear."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from semcache.cache import SemanticCache
from semcache.config import Config
from semcache.telemetry import report

app = typer.Typer(add_completion=False, help="Semantic cache for LLM calls.")
console = Console()


@app.callback()
def _configure() -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("SEMCACHE_LOG_LEVEL", "WARNING"))


def _redis_index(url: str):
    try:
        from semcache.index.redis import RedisIndex

        return RedisIndex(url)
    except Exception as e:  # redis missing or unreachable
        console.print(f"[yellow]Redis backend unavailable:[/yellow] {e}")
        return None


@app.command()
def bench(
    file: str = typer.Option(..., "--file", help='JSONL with {"prompt": ...} per line'),
    threshold: float = typer.Option(0.95, "--threshold"),
) -> None:
    """Run a prompt stream through the cache and print hit-rate + savings."""
    cache = SemanticCache(Config(threshold=threshold))
    lines = Path(file).read_text(encoding="utf-8").strip().splitlines()

    calls = {"n": 0}

    async def call_fn(p: str) -> str:
        calls["n"] += 1
        return f"answer::{p}"

    async def run() -> None:
        for line in lines:
            await cache.get_or_call(json.loads(line)["prompt"], call_fn)

    asyncio.run(run())
    rep = report(cache)
    table = Table(title="Semantic Cache — Bench", header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k in ("lookups", "hits", "misses", "exact_hits", "semantic_hits", "near_misses", "size"):
        table.add_row(k, str(rep[k]))
    table.add_row("hit_rate", f"[bold green]{rep['hit_rate'] * 100:.1f}%[/bold green]")
    table.add_row("saved_cost", f"${rep['saved_cost']:.4f}")
    table.add_row("LLM calls made", str(calls["n"]))
    console.print(table)


@app.command()
def stats(
    namespace: str = typer.Option("default", "--namespace"),
    redis_url: str = typer.Option("redis://localhost:6379/0", "--redis-url"),
) -> None:
    """Show entry counts in the (persistent) Redis backend."""
    idx = _redis_index(redis_url)
    if idx is None:
        raise typer.Exit(0)
    console.print(f"namespace [bold]{namespace}[/bold]: {len(idx.entries(namespace))} entries "
                  f"(total {len(idx)})")


@app.command()
def clear(
    namespace: str = typer.Option(None, "--namespace", help="omit to clear all"),
    redis_url: str = typer.Option("redis://localhost:6379/0", "--redis-url"),
) -> None:
    """Clear entries from the Redis backend."""
    idx = _redis_index(redis_url)
    if idx is None:
        raise typer.Exit(0)
    idx.clear(namespace)
    console.print(f"[green]cleared[/green] {namespace or 'all namespaces'}")


if __name__ == "__main__":
    app()
