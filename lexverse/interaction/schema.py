from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class InteractionState(BaseModel):
    step_count: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    dialog_history: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ParticipantAction(BaseModel):
    actor_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentResult(BaseModel):
    status: Literal["completed", "failed", "timeout", "invalid"]
    answer: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    dialog_history: list[dict[str, Any]] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    raw_output: dict[str, Any] = Field(default_factory=dict)
