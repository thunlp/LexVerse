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

"""Trajectory model for ATIF (Agent Trajectory Interchange Format)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from lexverse.models.trajectories.agent import Agent
from lexverse.models.trajectories.final_metrics import FinalMetrics
from lexverse.models.trajectories.step import Step


class Trajectory(BaseModel):
    """Agent Trajectory in ATIF (Agent Trajectory Interchange Format)."""

    schema_version: Literal[
        "ATIF-v1.0",
        "ATIF-v1.1",
        "ATIF-v1.2",
        "ATIF-v1.3",
        "ATIF-v1.4",
        "ATIF-v1.5",
        "ATIF-v1.6",
        "ATIF-v1.7",
    ] = Field(
        default="ATIF-v1.7",
        description="String defining ATIF compatibility",
    )
    session_id: str | None = Field(
        default=None,
        description="Identifier for the agent run this trajectory belongs to.",
    )
    trajectory_id: str | None = Field(
        default=None,
        description="Canonical per-trajectory-document identifier, distinct from `session_id`.",
    )
    agent: Agent = Field(
        default=...,
        description="Object specifying the agent configuration",
    )
    steps: list[Step] = Field(
        default=...,
        min_length=1,
        description="Array of step objects representing the complete interaction history",
    )
    notes: str | None = Field(
        default=None,
        description="Custom information, design notes, or explanations",
    )
    final_metrics: FinalMetrics | None = Field(
        default=None,
        description="Summary metrics for the entire trajectory",
    )
    continued_trajectory_ref: str | None = Field(
        default=None,
        description="Reference to the continuation trajectory file if this trajectory is continued in another file",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom root-level metadata",
    )
    subagent_trajectories: list["Trajectory"] | None = Field(
        default=None,
        description="Array of embedded subagent trajectories.",
    )

    model_config = {"extra": "forbid"}

    def to_json_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(exclude_none=exclude_none, mode="json")

    @model_validator(mode="after")
    def validate_step_ids(self) -> "Trajectory":
        for i, step in enumerate(self.steps):
            expected_step_id = i + 1
            if step.step_id != expected_step_id:
                raise ValueError(
                    f"steps[{i}].step_id: expected {expected_step_id} "
                    f"(sequential from 1), got {step.step_id}"
                )
        return self

    @model_validator(mode="after")
    def validate_embedded_subagent_trajectory_ids(self) -> "Trajectory":
        if not self.subagent_trajectories:
            return self
        seen: set[str] = set()
        for i, sub in enumerate(self.subagent_trajectories):
            if sub.trajectory_id is None:
                raise ValueError(
                    f"subagent_trajectories[{i}].trajectory_id is required "
                    f"for embedded subagents "
                    f"(agent.name={sub.agent.name!r}, "
                    f"session_id={sub.session_id!r})"
                )
            if sub.trajectory_id in seen:
                raise ValueError(
                    f"subagent_trajectories[{i}].trajectory_id "
                    f"{sub.trajectory_id!r} is not unique within "
                    f"subagent_trajectories"
                )
            seen.add(sub.trajectory_id)
        return self

    @model_validator(mode="after")
    def validate_tool_call_references(self) -> "Trajectory":
        for step in self.steps:
            if step.observation is None:
                continue
            tool_call_ids = set()
            if step.tool_calls:
                tool_call_ids = {tc.tool_call_id for tc in step.tool_calls}
            for result in step.observation.results:
                if result.source_call_id is not None:
                    if result.source_call_id not in tool_call_ids:
                        raise ValueError(
                            f"Observation result references source_call_id "
                            f"'{result.source_call_id}' which is not found in "
                            f"step {step.step_id}'s tool_calls"
                        )
        return self

    def has_multimodal_content(self) -> bool:
        for step in self.steps:
            if isinstance(step.message, list):
                for part in step.message:
                    if part.type == "image":
                        return True
            if step.observation:
                for result in step.observation.results:
                    if isinstance(result.content, list):
                        for part in result.content:
                            if part.type == "image":
                                return True
        return False
