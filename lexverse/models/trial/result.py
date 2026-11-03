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

from typing import Any, Literal
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    provider: str | None = None


class AgentInfo(BaseModel):
    name: str
    version: str
    model_info: ModelInfo | None = None


class EnvironmentResult(BaseModel):
    status: Literal["completed", "failed", "timeout", "invalid"]
    answer: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    dialog_history: list[dict[str, Any]] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    raw_output: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ExecutionResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_sec: float
    timed_out: bool = False
    artifacts: dict[str, Path] = field(default_factory=dict)
