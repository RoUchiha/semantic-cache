"""CLI smoke tests: bench works offline; stats/clear degrade gracefully."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from semcache.cli import app

runner = CliRunner()


def test_bench_reports_hits(tmp_path):
    # Engineered duplicates -> guaranteed exact hits.
    prompts = ["reset my password", "reset my password", "what is the weather"]
    f = tmp_path / "p.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": p}) for p in prompts), encoding="utf-8")
    result = runner.invoke(app, ["bench", "--file", str(f), "--threshold", "0.95"])
    assert result.exit_code == 0
    assert "hit_rate" in result.stdout
    assert "LLM calls made" in result.stdout


def test_stats_without_redis_is_graceful():
    # Point at an unreachable Redis; command should exit cleanly with a message.
    result = runner.invoke(app, ["stats", "--redis-url", "redis://127.0.0.1:6390/0"])
    assert result.exit_code == 0
    assert "unavailable" in result.stdout.lower()
