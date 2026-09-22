"""Bound concurrent work and retry each item independently."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from lexverse.runtime.errors import TrialError

_T = TypeVar("_T")
_R = TypeVar("_R")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    wait_multiplier: float = 2.0
    retry_on: tuple[type[BaseException], ...] = (TrialError,)

    def backoff(self, attempt_index: int) -> float:
        delay = self.min_wait_sec * (self.wait_multiplier ** attempt_index)
        return min(delay, self.max_wait_sec)


class Scheduler(Generic[_T, _R]):
    def __init__(self, n_concurrent: int, retry: RetryPolicy | None = None):
        if n_concurrent < 1:
            raise ValueError("n_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(n_concurrent)
        self._retry = retry or RetryPolicy()

    async def run(
        self,
        items: list[_T],
        worker: Callable[[_T], Awaitable[_R]],
    ) -> list[_R | BaseException]:
        """Execute worker(item) for each item; return results in input order.

        Failures are captured as BaseException entries — callers decide
        how to distinguish success from failure per item.
        """
        return await asyncio.gather(
            *(self._one(item, worker) for item in items),
            return_exceptions=True,
        )

    async def _one(
        self,
        item: _T,
        worker: Callable[[_T], Awaitable[_R]],
    ) -> _R:
        async with self._sem:
            last_exc: BaseException | None = None
            for attempt in range(self._retry.max_attempts):
                try:
                    return await worker(item)
                except self._retry.retry_on as exc:
                    last_exc = exc
                    if attempt + 1 >= self._retry.max_attempts:
                        raise
                    await asyncio.sleep(self._retry.backoff(attempt))
            # unreachable — the loop either returns or raises above
            assert last_exc is not None
            raise last_exc
