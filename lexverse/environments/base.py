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

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
import logging
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lexverse.models.trial import (
    EnvironmentType,
    ResourceMode,
    TaskEnvConfig,
    TaskOS,
    TrialPaths,
)
from lexverse.utils.env import resolve_env_vars
from lexverse.utils.logger import logger as global_logger


class ExecResult(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    return_code: int


class EnvironmentCapabilities(BaseModel):
    gpus: bool = False
    tpus: bool = False
    disable_internet: bool = False
    network_allowlist: bool = False
    windows: bool = False
    mounted: bool = False
    docker_compose: bool = False


class EnvironmentResourceCapabilities(BaseModel):
    cpu_limit: bool = False
    cpu_request: bool = False
    memory_limit: bool = False
    memory_request: bool = False


class ExecutionEnvironment(ABC):
    """
    Harbor-compatible execution environment interface for general agents.

    Provides standardized primitives (start, stop, exec, upload_file, upload_dir,
    download_file, download_dir) so any installed or custom agent can operate
    transparently in a local host process or a Docker container.
    """

    environment_dir: Path
    environment_name: str
    session_id: str
    trial_paths: TrialPaths
    task_env_config: TaskEnvConfig
    logger: logging.Logger

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: TaskEnvConfig,
        keep_containers: bool = False,
        skills_root: Path | None = None,
        logger: logging.Logger | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        suppress_override_warnings: bool = False,
        cpu_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        memory_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        network_policy: Any | None = None,
        *args,
        **kwargs,
    ):
        self.environment_dir = environment_dir
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.skills_root = skills_root or trial_paths.skills_dir
        self._execution_depth = 0
        self.task_env_config = task_env_config
        self._override_cpus = override_cpus
        self._override_memory_mb = override_memory_mb
        self._override_storage_mb = override_storage_mb
        self._override_gpus = override_gpus
        self._suppress_override_warnings = suppress_override_warnings
        self._cpu_enforcement_policy = cpu_enforcement_policy
        self._memory_enforcement_policy = memory_enforcement_policy
        self._network_policy = network_policy
        self.keep_containers = keep_containers

        self.logger = (logger or global_logger).getChild(self.__class__.__name__)
        self._maybe_override_task_env_config()
        self._validate_definition()
        self._persistent_env: dict[str, str] = (
            resolve_env_vars(self.task_env_config.env)
            if self.task_env_config.env
            else {}
        )
        self._scoped_env_layers: list[dict[str, str]] = []

    @asynccontextmanager
    async def agent_execution(self):
        outer = self._execution_depth == 0
        if outer:
            await self._begin_agent_execution()
        self._execution_depth += 1
        try:
            yield
        finally:
            self._execution_depth -= 1
            if outer:
                await self._end_agent_execution()

    async def _begin_agent_execution(self):
        raise RuntimeError("This environment does not implement task filesystem isolation")

    async def _end_agent_execution(self):
        return None

    def knowledge_read_paths(self) -> list[Path]:
        manifest = self.trial_paths.trial_dir / ".internal/knowledge_indexes.json"
        if not manifest.is_file():
            return []
        data = json.loads(manifest.read_text())
        paths = [manifest]
        for record in data.get("knowledge", []):
            paths.append(Path(record["path"]))
            model = record.get("retrieval", {}).get("embedding", {}).get("model")
            if model and Path(model).is_dir():
                paths.append(Path(model))
        if data.get("keyword_cache"):
            paths.append(Path(data["keyword_cache"]))
        return paths

    def instruction_paths(self, instruction: str) -> str:
        return instruction

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        if not self._persistent_env and not self._scoped_env_layers and not env:
            return None
        merged = dict(self._persistent_env)
        for layer in self._scoped_env_layers:
            merged.update(layer)
        if env:
            merged.update(env)
        return merged

    @contextmanager
    def scoped_exec_env(self, env: dict[str, str] | None) -> Iterator[None]:
        if not env:
            yield
            return
        self._scoped_env_layers.append(dict(env))
        try:
            yield
        finally:
            self._scoped_env_layers.pop()

    def _resolve_workdir(self, cwd: str | None) -> str | None:
        return cwd or self.task_env_config.workdir or str(self.trial_paths.workspace_dir)

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        if user is not None:
            return user
        return self.default_user

    def _maybe_override_task_env_config(self) -> None:
        if self._override_cpus is not None:
            self.task_env_config.cpus = self._override_cpus
        if self._override_memory_mb is not None:
            self.task_env_config.memory_mb = self._override_memory_mb
        if self._override_storage_mb is not None:
            self.task_env_config.storage_mb = self._override_storage_mb
        if self._override_gpus is not None:
            self.task_env_config.gpus = self._override_gpus

    @staticmethod
    @abstractmethod
    def type() -> EnvironmentType:
        """The environment type."""

    @property
    def os(self) -> TaskOS:
        return self.task_env_config.os

    @property
    @abstractmethod
    def is_mounted(self) -> bool:
        """Whether the environment mounts the logging directories."""

    @property
    @abstractmethod
    def capabilities(self) -> EnvironmentCapabilities:
        """Feature capabilities supported by this environment implementation."""

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    def default_user(self) -> str | int | None:
        return None

    @abstractmethod
    def _validate_definition(self) -> None:
        """Validate the necessary files exist in the environment directory."""

    @abstractmethod
    async def start(self, force_build: bool = False) -> None:
        """Starts the environment and optionally forces a build."""

    @abstractmethod
    async def stop(self, delete: bool = True) -> None:
        """Stops the environment and optionally deletes it."""

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Uploads a local file to the environment."""

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Uploads a local directory to the environment."""

    @abstractmethod
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Downloads a file from the environment to the local machine."""

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Downloads a directory from the environment to the local machine."""

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Executes a command in the environment."""

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(f"test -d {path}", timeout_sec=10, user=user)
        return result.return_code == 0

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        result = await self.exec(f"test -f {path}", timeout_sec=10, user=user)
        return result.return_code == 0

    def map_virtual_path(self, path: str) -> Path:
        """Map a container/virtual path like /inputs, /skills, /workspace, /artifacts, /logs to host path."""
        return Path(path)

