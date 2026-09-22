"""In-process environment for direct-response benchmarks."""
from __future__ import annotations

from pathlib import Path

from lexverse.interaction.engine import InteractionEngine
from lexverse.interaction.participants.base import Participant
from lexverse.interaction.policies.base import InteractionPolicy
from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask

from .base import ExecutionEnvironment


class DirectResponseEnvironment(ExecutionEnvironment):
    name = "direct_response"

    def __init__(
        self,
        *,
        policy: InteractionPolicy,
        participants: dict[str, Participant],
        context: dict | None = None,
        engine: InteractionEngine | None = None,
    ) -> None:
        self.policy = policy
        self.participants = participants
        self.context = context or {}
        self.engine = engine or InteractionEngine()

    async def prepare(self, context: dict | None = None) -> None:
        if context:
            self.context.update(context)

    async def run(self, task: LexVerseTask, work_dir: Path) -> EnvironmentResult:
        del work_dir
        return await self.engine.execute(
            task, self.policy, self.participants, context=self.context
        )
