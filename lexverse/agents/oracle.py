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

from pathlib import Path
import shlex
from typing import override

from lexverse.agents.base import BaseAgent
from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.models.trial import EnvironmentPaths


class OracleAgent(BaseAgent):
    """
    Reference solution agent that uploads `solution/` into `/solution` and executes
    `solve.sh` inside the sandbox environment.
    """

    _ORACLE_LOG_FILE = "oracle.txt"
    SUPPORTS_WINDOWS: bool = True

    @staticmethod
    @override
    def name() -> str:
        return AgentName.ORACLE.value

    def __init__(
        self,
        logs_dir: Path,
        task_dir: Path | None = None,
        solution_dir: Path | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        agent_timeout_sec: float | None = None,
        **kwargs,
    ):
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            extra_env=extra_env,
            **kwargs,
        )
        self._task_dir = Path(task_dir) if task_dir is not None else None
        self._solution_dir = (
            Path(solution_dir)
            if solution_dir is not None
            else (self._task_dir / "solution" if self._task_dir else None)
        )
        self._agent_timeout_sec = agent_timeout_sec

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: ExecutionEnvironment) -> None:
        if self._solution_dir is None or not (self._solution_dir / "solve.sh").is_file():
            raise FileNotFoundError(f"Solution script not found: {self._solution_dir}")
        await environment.upload_dir(self._solution_dir, "/runtime/oracle/solution")

    @override
    async def run(
        self,
        instruction: str,
        environment: ExecutionEnvironment,
        context: AgentContext,
    ) -> None:
        env_paths = EnvironmentPaths.for_os(environment.os)
        container_solve_path = "/runtime/oracle/solution/solve.sh"
        container_oracle_log_path = str(env_paths.agent_dir / self._ORACLE_LOG_FILE)

        timeout_sec = int(self._agent_timeout_sec) if self._agent_timeout_sec else None
        command = (
            f"bash {shlex.quote(container_solve_path)} "
            f"> {shlex.quote(container_oracle_log_path)} 2>&1"
        )
        result = await environment.exec(
            command=command,
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0:
            exit_code_path = self.logs_dir / "exit-code.txt"
            exit_code_path.write_text(str(result.return_code), encoding="utf-8")
