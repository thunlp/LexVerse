from __future__ import annotations

from pathlib import Path

from lexverse.interaction.engine import InteractionEngine
from lexverse.interaction.participants import Participant
from lexverse.interaction.policies import InteractionPolicy
from lexverse.interaction.schema import EnvironmentResult
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
        context = {**self.context, "work_dir": str(work_dir), "task_id": task.id,
                   "task_type": task.source.task_type}
        result = await self.engine.execute(
            task, self.policy, self.participants, context=context
        )
        metadata = result.final_state.get("data", {}).get("capabilities")
        if metadata:
            result.artifacts.update(capability_result=metadata["result_ref"], capability_trace=metadata["trace_ref"])
            result.trace.append(metadata)
        return result
