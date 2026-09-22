from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import ModelResponse


@dataclass
class ScriptedProvider:
    responder: Callable[[list[dict[str, Any]]], str]
    name: str = "scripted"

    async def generate(self, messages, *, config=None) -> ModelResponse:
        return ModelResponse(content=self.responder(messages), model=self.name)
