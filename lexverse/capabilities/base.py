from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class CapabilitySelection(BaseModel):
    enabled: bool = False
    knowledge: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    workflows: list[str] = Field(default_factory=list)
    mcp: list[str] = Field(default_factory=list)


class Capability(Protocol):
    name: str

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]: ...

