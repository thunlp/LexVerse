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

"""Subagent trajectory reference model for ATIF trajectories."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class SubagentTrajectoryRef(BaseModel):
    """Reference to a delegated subagent trajectory."""

    trajectory_id: str | None = Field(
        default=None,
        description=(
            "Canonical identifier of the delegated subagent trajectory. "
            "Matches `Trajectory.trajectory_id` of an entry in the parent's "
            "`subagent_trajectories` array and is the resolution key for "
            "embedded references. Added in ATIF-v1.7."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Run identity of the delegated subagent trajectory. Informational "
            "only: recorded so consumers can correlate this ref back to the "
            "subagent's run for debug / search / display purposes."
        ),
    )
    trajectory_path: str | None = Field(
        default=None,
        description=(
            "Location of the complete subagent trajectory as an external "
            "file (file path, S3 URL, database reference, etc.)."
        ),
    )
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Custom metadata about the subagent execution",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_is_resolvable(self) -> "SubagentTrajectoryRef":
        if self.trajectory_id is None and self.trajectory_path is None:
            raise ValueError(
                "SubagentTrajectoryRef must be resolvable: set either "
                "`trajectory_id` (for embedded references) or "
                "`trajectory_path` (for external-file references)."
            )
        return self
