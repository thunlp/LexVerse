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
import logging
from pathlib import Path

from pydantic import BaseModel

from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.models.trial import MCPServerConfig
from lexverse.utils.logger import logger as global_logger


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

    SUPPORTS_ATIF: bool = False
    SUPPORTS_RESUME: bool = False
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        memory_dir: str | None = None,
        session_id: str | None = None,
        *args,
        **kwargs,
    ):
        self.logs_dir = logs_dir
        self.model_name = model_name
        self.logger = (logger or global_logger).getChild(self.__class__.__name__)
        self.mcp_servers = mcp_servers or []
        if skills_dir not in {None, "/skills"}:
            raise ValueError("Task Skills must use the single /skills directory")
        self.skills_dir = skills_dir
        self.memory_dir = memory_dir
        self.session_id = session_id

        self._extra_env: dict[str, str] = {}
        if extra_env:
            self._extra_env.update(extra_env)

        self._parsed_model_provider: str | None = None
        self._parsed_model_name: str | None = None
        if model_name:
            if "/" in model_name:
                self._parsed_model_provider, self._parsed_model_name = model_name.split(
                    "/", maxsplit=1
                )
            else:
                self._parsed_model_name = model_name

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
