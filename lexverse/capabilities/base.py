from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    from lexverse.config import ModelConfig
    from lexverse.tasks.user import AgentLimits, OutputContract, RetrievalConfig, TaskCapabilities


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
    environment: str = "docker"
    context_window_tokens: int | None = None
    skills_root: Path | None = None


def execution_conditions(identity):
    return {key: value for key, value in identity.items() if key not in {"code", "code_commit"}}
