"""@cached — wrap an async LLM-call function with the semantic cache."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps

from semcache.cache import SemanticCache


def cached(cache: SemanticCache, namespace: str = "default"):
    """Decorate an async ``fn(prompt) -> str`` so equivalent prompts are served
    from cache instead of re-invoking the function."""

    def decorator(fn: Callable[[str], Awaitable[str]]):
        @wraps(fn)
        async def wrapper(prompt: str) -> str:
            response, _hit = await cache.get_or_call(prompt, fn, namespace=namespace)
            return response

        return wrapper

    return decorator
