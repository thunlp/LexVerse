# Copyright 2025-2026 Alex Shaw and the Harbor Framework Team
# Portions of this file are derived from Harbor (https://github.com/laude-institute/harbor, v0.20.0).
# Modified by the LexVerse Contributors for LexVerse package integration and legal capability support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step model for ATIF trajectories."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from lexverse.models.trajectories.content import ContentPart
from lexverse.models.trajectories.metrics import Metrics
from lexverse.models.trajectories.observation import Observation
from lexverse.models.trajectories.tool_call import ToolCall


class Step(BaseModel):
    """A single step in the trajectory."""

    step_id: int = Field(
        default=...,
        ge=1,
        description="Ordinal index of the turn (starting from 1)",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp indicating when this step occurred",
    )
    source: Literal["system", "user", "agent"] = Field(
        default=...,
        description="The originator of this step",
    )
    model_name: str | None = Field(
        default=None,
        description="The specific LLM model used for this turn.",
    )
    reasoning_effort: str | float | None = Field(
        default=None,
        description="Qualitative or quantitative measure of effort",
    )
    message: str | list[ContentPart] = Field(
        default=...,
        description="The dialogue message.",
    )
    reasoning_content: str | None = Field(
        default=None,
        description="The agent's explicit internal reasoning",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Array of structured objects for the agent's actions",
    )
    observation: Observation | None = Field(
        default=None,
        description="Environment feedback/result after actions or system events",
    )
    metrics: Metrics | None = Field(
        default=None,
        description="LLM operational and confidence data for this step",
    )
    is_copied_context: bool | None = Field(
        default=None,
        description="Indicates whether this step was copied from a previous trajectory.",
    )
    llm_call_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of LLM inferences this step represents.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom step-level metadata",
    )

    model_config = {"extra": "forbid"}

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(f"Invalid ISO 8601 timestamp: {e}")
        return v

    @model_validator(mode="after")
    def validate_agent_only_fields(self) -> "Step":
        if self.source != "agent":
            agent_only_fields = [
                "model_name",
                "reasoning_effort",
                "reasoning_content",
                "tool_calls",
                "metrics",
            ]
            for field in agent_only_fields:
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"Field '{field}' is only applicable when source is 'agent', "
                        f"but source is '{self.source}'"
                    )
        return self

    @model_validator(mode="after")
    def validate_llm_call_count_zero_fields(self) -> "Step":
        if self.llm_call_count == 0 and self.source == "agent":
            for field in ("metrics", "reasoning_content"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"Field '{field}' must be absent when llm_call_count is 0 "
                        f"(deterministic dispatch on a 'source: agent' step)"
                    )
        return self
