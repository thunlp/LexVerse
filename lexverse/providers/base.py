"""Benchmark-neutral provider protocol and response model."""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    content: str
    model: str | None = None
    # OpenAI-compatible providers may add nested token details, latency
    # checkpoints, or vendor-specific metadata alongside integer totals.
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_sec: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


class Provider(Protocol):
    name: str

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        config: dict[str, Any] | None = None,
    ) -> ModelResponse: ...
