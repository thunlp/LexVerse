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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import logging
from pathlib import Path

from pydantic import BaseModel

from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.models.task.environment import MCPServerConfig
from lexverse.utils.logger import logger as global_logger
from lexverse.agents.model_connection import ModelConnectionSpec, ResolvedModelConnection, resolve_model_connection


class AgentSetupInfo(BaseModel):
    name: str
    version: str
    model_info: dict | None = None


class BaseAgent(ABC):
    """
    Harbor-compatible base class for all general agents in LexVerse.

    Every agent (whether an external CLI agent like Claude Code / Codex / OpenHands /
    Qwen Code / Gemini CLI, the built-in LexVerse LangGraph DeepAgent, or a custom
    user agent) implements `setup(environment)` and `run(instruction, environment, context)`.
    """

    MODEL_CONNECTION: ModelConnectionSpec | None = ModelConnectionSpec()
    SUPPORTS_NETWORK_ALLOWLIST: bool = False

    SUPPORTS_ATIF: bool = False
    SUPPORTS_RESUME: bool = False
    SUPPORTS_WINDOWS: bool = False
    SUPPORTS_SKILLS: bool = False
    MCP_TRANSPORTS: frozenset[str] = frozenset()
    SUPPORTS_MCP_HEADERS: bool = False
    SUPPORTS_MCP_ENV: bool = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        model_connection: ResolvedModelConnection | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        memory_dir: str | None = None,
        session_id: str | None = None,
        *args,
        **kwargs,
    ):
        self.logs_dir = logs_dir
        import os
        self.model_connection = model_connection or self.resolve_connection(
            model_name, env={**os.environ, **(extra_env or {})})
        self.model_name = self.model_connection.model_name
        self.logger = (logger or global_logger).getChild(self.__class__.__name__)
        self.mcp_servers = mcp_servers or []
        if skills_dir and not self.SUPPORTS_SKILLS:
            raise ValueError(f"{self.name()} does not support Skills")
        for server in self.mcp_servers:
            if server.transport not in self.MCP_TRANSPORTS:
                raise ValueError(f"{self.name()} does not support MCP transport {server.transport}")
            if server.headers and not self.SUPPORTS_MCP_HEADERS:
                raise ValueError(f"{self.name()} does not support MCP headers")
            if server.env and not self.SUPPORTS_MCP_ENV:
                raise ValueError(f"{self.name()} does not support MCP process environment variables")
        if skills_dir not in {None, "/skills"}:
            raise ValueError("Task Skills must use the single /skills directory")
        self.skills_dir = skills_dir
        self.memory_dir = memory_dir
        self.session_id = session_id

        self._extra_env = {**(extra_env or {}), **self.model_connection.env}
        self._parsed_model_provider = self.model_connection.provider
        self._parsed_model_name = self.model_connection.model

    @classmethod
    def resolve_connection(cls, model_name, *, profile=None, env=None, use_api_credentials=True):
        return resolve_model_connection(model_name, cls.MODEL_CONNECTION, profile=profile, env=env,
                                        use_api_credentials=use_api_credentials)

    @classmethod
    def required_endpoints(cls, connection, *, env=None):
        return [connection.base_url] if connection.base_url else []

    def read_final_answer(self) -> str | None:
        """Return native terminal text; file-only agents may return None."""
        return None

    def to_agent_info(self) -> AgentSetupInfo:
        return AgentSetupInfo(
            name=self.name(),
            version=self.version() or "unknown",
            model_info=(
                {
                    "name": self._parsed_model_name,
                    "provider": self._parsed_model_provider or "unknown",
                }
                if self._parsed_model_name
                else None
            ),
        )

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the agent."""

    @abstractmethod
    def version(self) -> str | None:
        """The version of the agent."""

    @abstractmethod
    async def setup(self, environment: ExecutionEnvironment) -> None:
        """Run commands to setup the agent & its tools in the environment."""

    @abstractmethod
    async def run(
        self,
        instruction: str,
        environment: ExecutionEnvironment,
        context: AgentContext,
    ) -> None:
        """Runs the agent in the environment."""


if TYPE_CHECKING:
    from lexverse.config import ModelConfig
    from lexverse.models.task.config import AgentLimits, RetrievalConfig
    from lexverse.capabilities.resources import ResourceSelection


@dataclass
class AgentExecutionSpec:
    run_dir: Path
    thread_id: str
    messages: list[dict]
    model: ModelConfig
    selection: ResourceSelection
    retrieval: RetrievalConfig
    limits: AgentLimits
    interactive: bool
    mode: str = "user_task"
    environment: str = "docker"
    context_window_tokens: int | None = None
    skills_root: Path | None = None
    network_mode: str = "public"
    allowed_hosts: list[str] = field(default_factory=list)
    environment_config: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    verifier: dict | None = None
    task_skills: list[str] = field(default_factory=list)

    def to_dict(self):
        from dataclasses import asdict
        return {
            **{name: getattr(self, name) for name in ("thread_id", "messages", "interactive", "mode",
                "environment", "context_window_tokens", "network_mode", "allowed_hosts",
                "environment_config", "artifacts", "verifier", "task_skills")},
            "run_dir": str(self.run_dir), "skills_root": str(self.skills_root) if self.skills_root else None,
            "model": asdict(self.model),
            **{name: getattr(self, name).model_dump(mode="json") for name in
               ("selection", "retrieval", "limits")},
        }

    @classmethod
    def from_dict(cls, value):
        from lexverse.config import ModelConfig
        from lexverse.models.task.config import AgentLimits, RetrievalConfig
        from lexverse.capabilities.resources import ResourceSelection
        return cls(**{**value, "run_dir": Path(value["run_dir"]),
                      "skills_root": Path(value["skills_root"]) if value.get("skills_root") else None,
                      "model": ModelConfig(**value["model"]),
                      "selection": ResourceSelection.model_validate(value["selection"]),
                      "retrieval": RetrievalConfig.model_validate(value["retrieval"]),
                      "limits": AgentLimits.model_validate(value["limits"])})


def execution_identity(spec, *, agent_name="vanilla"):
    from dataclasses import asdict
    import hashlib
    import importlib.metadata
    import subprocess
    from lexverse.capabilities.base import identity

    package = Path(__file__).resolve().parents[1]
    files = [*package.joinpath("agents").rglob("*.py"), *package.joinpath("capabilities").rglob("*.py"),
             package / "cli/tasks.py", package / "tasks/user.py", package / "llms/api.py",
             package / "models/trial/paths.py", package / "agents/base.py", package / "config.py"]
    files += [*package.joinpath("trial").glob("*.py"), *package.joinpath("models").rglob("*.py"),
              package / "tasks/runner.py", package / "job.py"]
    files += [*package.joinpath("environments").glob("*.py"),
              *(path for path in package.joinpath("environments/egress").rglob("*") if path.is_file())]
    versions = {name: importlib.metadata.version(name) for name in (
        "deepagents", "langchain", "langchain-core", "langchain-openai", "langgraph",
        "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "fastmcp", "pydantic",
    )} if agent_name == "vanilla" else {}
    try:
        revision = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=5, check=False)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    configuration = {"agent": agent_name, "environment": spec.environment, "model": asdict(spec.model), "mode": spec.mode,
            "selection": spec.selection.model_dump(mode="json"), "retrieval": spec.retrieval.model_dump(),
            "limits": spec.limits.model_dump(), "artifacts": spec.artifacts, "task_skills": spec.task_skills,
            "environment_config": spec.environment_config, "verifier": spec.verifier,
            "context_window_tokens": spec.context_window_tokens,
            "network_mode": spec.network_mode, "allowed_hosts": spec.allowed_hosts}
    return {"configuration_hash": identity(configuration), "messages_hash": identity(spec.messages),
            "code_commit": commit,
            "code": identity({p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}),
            "versions": versions}
