from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InteractionState(BaseModel):
    step_count: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    dialog_history: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

