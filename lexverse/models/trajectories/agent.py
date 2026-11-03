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

"""Agent configuration model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field


class Agent(BaseModel):
    """Agent configuration."""

    name: str = Field(
        default=...,
        description="The name of the agent system",
    )
    version: str = Field(
        default=...,
        description="The version identifier of the agent system",
    )
    model_name: str | None = Field(
        default=None,
        description="Default LLM model used for this trajectory",
    )
    tool_definitions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Array of tool/function definitions available to the agent. Each element follows OpenAI's function calling schema.",
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom agent configuration details",
    )

    model_config = {"extra": "forbid"}
