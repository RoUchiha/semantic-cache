"""Config: threshold, TTL, eviction, backend, savings parameters (YAML -> Pydantic)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    # Conservative by default: a false hit serves a wrong answer, so err high.
    threshold: float = 0.95
    ttl_seconds: float = 3600.0           # <= 0 disables expiry
    max_entries: int = 1000
    eviction: str = "lru"                 # "lru" | "none"
    backend: str = "memory"               # "memory" | "redis"
    normalize_case: bool = True
    near_miss_margin: float = 0.05        # log matches within this of the threshold

    # For savings telemetry (estimates).
    cost_per_call: float = 0.01
    latency_per_call_ms: float = 800.0

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)
