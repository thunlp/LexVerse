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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class VerifierProvenance(BaseModel):
    name: str
    version: str
    upstream_commit: str | None = None
    config_hash: str | None = None
    model: str | None = None


class VerifierResult(BaseModel):
    status: Literal["completed", "failed", "invalid", "not_applicable"]
    metrics: dict[str, float | int] = Field(default_factory=dict)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    provenance: VerifierProvenance

    @field_validator("metrics", mode="before")
    @classmethod
    def _reject_boolean_metrics(cls, metrics):
        if isinstance(metrics, dict) and any(isinstance(value, bool) for value in metrics.values()):
            raise ValueError("metrics must be numbers, not booleans")
        return metrics

    @field_validator("metrics")
    @classmethod
    def _finite_numeric_metrics(cls, metrics):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"metric {key!r} must be a finite number")
        return metrics


@dataclass
class RunVerificationResult:
    """Native scoring groups and optional independently scored trial results."""

    groups: list[dict[str, Any]] = field(default_factory=list)
    trials: dict[str, VerifierResult] = field(default_factory=dict)
    expected_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        scored = {task_id for group in self.groups for task_id in group["task_ids"]}
        scored -= {task_id for task_id, result in self.trials.items() if result.status != "completed"}
        missing = sorted(set(self.expected_task_ids) - scored)
        return {
            "status": "completed" if not missing else "incomplete",
            "expected_task_ids": self.expected_task_ids,
            "scored_task_ids": sorted(scored),
            "missing_task_ids": missing,
            "groups": self.groups,
            "trials": {key: value.model_dump(mode="json") for key, value in self.trials.items()},
        }

    def task_metrics(self) -> dict[str, dict[str, float]]:
        return {group["task_type"]: group["metrics"] for group in self.groups}

    @classmethod
    def from_dict(cls, data):
        for group in data["groups"]:
            if (not isinstance(group["task_type"], str) or not isinstance(group["task_ids"], list)
                    or not all(isinstance(key, str) for key in group["task_ids"])
                    or not isinstance(group["metrics"], dict)
                    or not all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                               for value in group["metrics"].values())):
                raise ValueError("invalid native scoring group")
        return cls(groups=data["groups"], expected_task_ids=data["expected_task_ids"],
                   trials={key: VerifierResult.model_validate(value) for key, value in data["trials"].items()})


class TaskVerifierResult(BaseModel):
    rewards: dict[str, float | int] | None = None
    return_code: int = 0
    stdout: str | None = None
    stderr: str | None = None


