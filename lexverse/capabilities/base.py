from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lexverse.config import ModelConfig
    from lexverse.tasks.user import AgentLimits, OutputContract, RetrievalConfig, TaskCapabilities


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


@dataclass
class AgentExecutionSpec:
    run_dir: Path
    thread_id: str
    messages: list[dict]
    model: ModelConfig
    selection: TaskCapabilities
    retrieval: RetrievalConfig
    limits: AgentLimits
    outputs: OutputContract
    interactive: bool
    mode: str = "user_task"
    context_window_tokens: int | None = None
    skills_root: Path | None = None


def execution_conditions(identity):
    return {key: value for key, value in identity.items() if key not in {"code", "code_commit"}}
