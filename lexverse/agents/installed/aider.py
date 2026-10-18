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

import os
import shlex
from typing import override

from lexverse.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from lexverse.environments.base import BaseEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName


class Aider(BaseInstalledAgent):
    """
    The Aider agent installs https://aider.chat/ into the task container and attempts to
    solve the task using Aider's scripting mode.
    """

    SUPPORTS_RESUME: bool = True

    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="--reasoning-effort",
            type="str",
        ),
        CliFlag(
            "thinking_tokens",
            cli="--thinking-tokens",
            type="int",
        ),
        CliFlag(
            "cache_prompts",
            cli="--cache-prompts",
            type="bool",
        ),
        CliFlag(
            "auto_lint",
            cli="--auto-lint",
            type="bool",
        ),
        CliFlag(
            "auto_test",
            cli="--auto-test",
            type="bool",
        ),
        CliFlag(
            "test_cmd",
            cli="--test-cmd",
            type="str",
        ),
        CliFlag(
            "stream",
            cli="--stream",
            type="bool",
        ),
        CliFlag(
            "map_tokens",
            cli="--map-tokens",
            type="int",
        ),
    ]

    @staticmethod
    @override
    def name() -> str:
        return AgentName.AIDER.value

    @override
    def get_version_command(self) -> str | None:
        return ". $HOME/.local/bin/env; aider --version"

    @override
    def parse_version(self, stdout: str) -> str:
        text = stdout.strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line.removeprefix("aider").strip()
        return text

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -LsSf https://aider.chat/install.sh | sh && "
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                "aider --version"
            ),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        pass

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        # Determine provider and API key from model name
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, model = self.model_name.split("/", 1)

        # Get API key based on provider
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not api_key:
            raise ValueError(f"No API key found for provider: {provider}")

        env = {
            "AIDER_API_KEY": f"{provider}={api_key}",
        }

        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""
        restore_flag = "--restore-chat-history " if self._resume else ""

        await self.exec_as_agent(
            environment,
            command=(
                ". $HOME/.local/bin/env; "
                "aider --yes "
                "--chat-history-file=/logs/agent/aider.chat.history.md "
                f"{restore_flag}{extra_flags}--model={model} "
                f"--message={escaped_instruction} "
                f"2>&1 | stdbuf -oL tee /logs/agent/aider.txt"
            ),
            env=env,
        )
