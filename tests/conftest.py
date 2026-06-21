"""Shared fixtures. Scripted vectors give precise control over cosine scores."""

from __future__ import annotations

import math

import pytest

from semcache.embed import MockEmbedder

# Vectors chosen so cosine(A,B)~0.995 (semantic hit @0.95) and cosine(A,C)=0 (miss).
# D is a near-miss: cosine(A,D)=0.92.
VECTORS = {
    "how do i reset my password": [1.0, 0.0],
    "how can i reset my password": [0.99, 0.1],          # ~0.995 vs A
    "what is the capital of france": [0.0, 1.0],         # 0 vs A
    "i want to change my password": [0.92, math.sqrt(1 - 0.92**2)],  # 0.92 vs A
}


@pytest.fixture
def embedder():
    return MockEmbedder(vectors=VECTORS, dim=8)


class Clock:
    """Controllable monotonic clock for TTL tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()
