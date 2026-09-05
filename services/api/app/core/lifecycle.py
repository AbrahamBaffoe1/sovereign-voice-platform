"""Concurrency-safe lifecycle primitives for expensive lazily loaded model resources."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncLazy(Generic[T]):
    """Concurrency-safe lazy resource loader.

    GPU models are expensive and multiple first requests can otherwise race and load
    duplicate copies into VRAM. The loader executes once and all waiters share it.
    """

    def __init__(self, factory: Callable[[], Awaitable[T]]) -> None:
        """Store the async factory without loading the resource and create the lock that will serialize
        the first materialization/reset transition."""
        self._factory = factory
        self._value: T | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        """Report whether the resource has already been materialized without triggering a load."""
        return self._value is not None

    async def get(self) -> T:
        """Return the cached resource or create it exactly once under an async lock. The second check
        inside the lock is required because another waiter may have completed loading while this
        coroutine was queued."""
        if self._value is not None:
            return self._value
        async with self._lock:
            if self._value is None:
                self._value = await self._factory()
        return self._value

    async def reset(self, disposer: Callable[[T], Awaitable[None]] | None = None) -> None:
        """Atomically detach the cached resource and optionally dispose it while holding the lifecycle
        lock, preventing a concurrent request from observing a half-reset model."""
        async with self._lock:
            value, self._value = self._value, None
            if value is not None and disposer is not None:
                await disposer(value)
