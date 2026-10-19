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

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TaskOS(str, Enum):
    LINUX = "linux"
    WINDOWS = "windows"


class EnvironmentType(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"


class ResourceMode(str, Enum):
    AUTO = "auto"
    LIMIT = "limit"
    REQUEST = "request"
    GUARANTEE = "guarantee"
    IGNORE = "ignore"


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["sse", "streamable-http", "stdio"] = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required when transport is '{self.transport}'.")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required when transport is 'stdio'.")
        return self


class TaskEnvConfig(BaseModel):
    os: TaskOS = TaskOS.LINUX
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = None
    tpus: int = 0
    tpu_type: str | None = None
    allow_internet: bool = True
    network_allowlist: list[str] | None = None
    docker_image: str | None = None
    workdir: str | None = None
    skills_dir: str | None = "/skills"
    memory_dir: str | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    build_timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    name: str
    provider: str | None = None


class AgentInfo(BaseModel):
    name: str
    version: str
    model_info: ModelInfo | None = None


class AgentConfig(BaseModel):
    name: str | None = "lexverse"
    import_path: str | None = None
    model_name: str | None = None
    override_timeout_sec: float | None = None
    override_setup_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentPaths:
    """
    Static paths used within the environment.
    """

    logs_dir: PurePosixPath = PurePosixPath("/logs")
    agent_dir: PurePosixPath = logs_dir / "agent"
    verifier_dir: PurePosixPath = logs_dir / "verifier"
    artifacts_dir: PurePosixPath = PurePosixPath("/artifacts")
    inputs_dir: PurePosixPath = PurePosixPath("/inputs")
    workspace_dir: PurePosixPath = PurePosixPath("/workspace")
    tests_dir: PurePosixPath = PurePosixPath("/tests")
    solution_dir: PurePosixPath = PurePosixPath("/solution")
    default_skills_dir: PurePosixPath = PurePosixPath("/skills")
    reward_text_path: PurePosixPath = verifier_dir / "reward.txt"
    reward_json_path: PurePosixPath = verifier_dir / "reward.json"

    @classmethod
    def for_windows(cls) -> "EnvironmentPaths":
        return cls._with_root(PurePosixPath("C:/"))

    @classmethod
    def for_os(cls, os: TaskOS) -> "EnvironmentPaths":
        if os == TaskOS.WINDOWS:
            return cls.for_windows()
        return cls()

    @classmethod
    def _with_root(cls, root: PurePosixPath) -> "EnvironmentPaths":
        logs_dir = root / "logs"
        verifier_dir = logs_dir / "verifier"
        return cls(
            logs_dir=logs_dir,
            agent_dir=logs_dir / "agent",
            verifier_dir=verifier_dir,
            artifacts_dir=root / "artifacts",
            inputs_dir=root / "inputs",
            workspace_dir=root / "workspace",
            tests_dir=root / "tests",
            solution_dir=root / "solution",
            default_skills_dir=root / "skills",
            reward_text_path=verifier_dir / "reward.txt",
            reward_json_path=verifier_dir / "reward.json",
        )


@dataclass(frozen=True)
class TrialPaths:
    """
    Host output directory paths for a trial / task run.
    """

    trial_dir: Path

    @property
    def _compact(self) -> bool:
        return (self.trial_dir / ".internal").exists()

    @property
    def _agent_fs(self) -> Path:
        return self.trial_dir / ".internal" / "agent_fs"

    def mkdir(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def chmod_dir(self) -> None:
        for path in (
            self.trial_dir,
            self.agent_dir,
            self.verifier_dir,
            self.artifacts_dir,
            self.workspace_dir,
        ):
            if path.exists():
                try:
                    path.chmod(0o777)
                except OSError:
                    pass

    @property
    def config_path(self) -> Path:
        return self.trial_dir / "config.json"

    @property
    def agent_dir(self) -> Path:
        return self.trial_dir / "agent"

    @property
    def artifacts_dir(self) -> Path:
        return self.trial_dir / "artifacts"

    @property
    def workspace_dir(self) -> Path:
        return self._agent_fs / "workspace" if self._compact else self.trial_dir / "workspace"

    @property
    def inputs_dir(self) -> Path:
        return self._agent_fs / "inputs" if self._compact else self.trial_dir / "inputs"

    @property
    def skills_dir(self) -> Path:
        return self._agent_fs / "skills" if self._compact else self.trial_dir / "skills"

    @property
    def verifier_dir(self) -> Path:
        return self.trial_dir / "verifier"

    @property
    def test_stdout_path(self) -> Path:
        return self.verifier_dir / "test-stdout.txt"

    @property
    def test_stderr_path(self) -> Path:
        return self.verifier_dir / "test-stderr.txt"

    @property
    def reward_text_path(self) -> Path:
        return self.verifier_dir / "reward.txt"

    @property
    def reward_json_path(self) -> Path:
        rewards = self.verifier_dir / "rewards.json"
        return rewards if rewards.is_file() else self.verifier_dir / "reward.json"

    @property
    def result_path(self) -> Path:
        return self.trial_dir / "result.json"

    @property
    def log_path(self) -> Path:
        return self.trial_dir / "trial.log"
