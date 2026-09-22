from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .base import ModelResponse
from lexverse.runtime.errors import ProviderError


@dataclass
class OpenAICompatibleProvider:
    name: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    default_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._client = None
        self.last_error: Exception | None = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai>=1.0 is required; install with `pip install lexverse[openai]`"
            ) from exc
        self._client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        return self._client

    async def generate(self, messages, *, config=None) -> ModelResponse:
        options = {**self.default_config, **(config or {})}
        try:
            response = await self._get_client().chat.completions.create(
                model=self.model, messages=messages, **options
            )
            content = response.choices[0].message.content
            if not content:
                raise ProviderError("model returned an empty response")
            usage = response.usage.model_dump() if response.usage else {}
            return ModelResponse(
                content=content, model=self.model, usage=usage,
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as exc:
            self.last_error = exc
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(
                f"provider {self.name!r} ({self.model}) failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
