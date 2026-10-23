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

from typing import override

from lexverse.agents.base import BaseAgent
from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName


class NopAgent(BaseAgent):
    SUPPORTS_WINDOWS: bool = True

    @staticmethod
    @override
    def name() -> str:
        return AgentName.NOP.value

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: ExecutionEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: ExecutionEnvironment,
        context: AgentContext,
    ) -> None:
        return None
