from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .base import ModelResponse
from lexverse.runtime.errors import ProviderError
from lexverse.config import ConfigError, ModelConfig, load_profile


@dataclass
class OpenAICompatibleProvider:
    name: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    default_config: dict[str, Any] = field(default_factory=dict)
    timeout_sec: float | None = None
    max_retries: int = 2

    def __post_init__(self) -> None:
        self._client = None
        self.last_error: Exception | None = None

    async def generate(self, messages, *, config=None) -> ModelResponse:
        options = {**self.default_config, **(config or {})}
        try:
            response = await self._get_client().chat.completions.create(
                model=self.model, messages=messages, **options
            )
            content = response.choices[0].message.content
            if not content:
                raise ProviderError("model returned an empty response")
            if "</think>" in content:
                content = content.split("</think>", 1)[1].lstrip()
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
            **({"timeout": self.timeout_sec} if self.timeout_sec is not None else {}),
            max_retries=self.max_retries,
        )
        return self._client


def create_chat_model(config: ModelConfig, *, context_window_tokens: int | None = None):
    if config.provider != "openai_compatible":
        raise ConfigError(
            "Direct weight backends are not yet supported for agent tool calling; "
            "use an OpenAI-compatible local service"
        )
    parameters = config.generation_parameters
    reserved = {"model", "model_name", "base_url", "openai_api_base", "api_key", "openai_api_key",
                "max_retries", "use_responses_api", "profile"}
    if reserved & parameters.keys():
        raise ConfigError("generation_parameters cannot override model connection or runtime settings")
    if context_window_tokens is not None and (type(context_window_tokens) is not int or context_window_tokens <= 0):
        raise ConfigError("context_window_tokens must be a positive integer")
    output_keys = [key for key in ("max_tokens", "max_completion_tokens") if parameters.get(key) is not None]
    if len(output_keys) > 1:
        raise ConfigError("Specify either max_tokens or max_completion_tokens, not both")
    output = parameters[output_keys[0]] if output_keys else None
    if output is not None and (type(output) is not int or output <= 0):
        raise ConfigError("Maximum output tokens must be a positive integer")
    from langchain_openai import ChatOpenAI

    profile = load_profile(config.profile)
    model = ChatOpenAI(
        model=config.name, base_url=profile.base_url, api_key=profile.api_key,
        use_responses_api=False, max_retries=0, **parameters,
    )
    capacity = context_window_tokens or (model.profile or {}).get("max_input_tokens")
    if not isinstance(capacity, int) or capacity <= 0:
        raise ConfigError("Unknown model context capacity; set context_window_tokens in task configuration")
    if output is not None and output >= capacity:
        raise ConfigError("Maximum output tokens must be smaller than context_window_tokens")
    model.profile = {**(model.profile or {}), "max_input_tokens": capacity - (output or 0)}
    return model
