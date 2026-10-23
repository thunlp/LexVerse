from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

from lexverse.interaction.participants import Participant
from lexverse.interaction.policies import InteractionPolicy
from lexverse.interaction.schema import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask


class InteractionEnvironment(ABC):
    name: str

    async def prepare(self, context: dict | None = None) -> None:
        return None

    @abstractmethod
    async def run(self, task: LexVerseTask, work_dir: Path) -> EnvironmentResult:
        pass

    async def close(self) -> None:
        return None


class StepLimitExceeded(RuntimeError):
    pass


class InteractionEngine:
    async def execute(
        self,
        task: LexVerseTask,
        policy: InteractionPolicy,
        participants: dict[str, Participant],
        *,
        context: dict | None = None,
    ) -> EnvironmentResult:
        state = policy.initialize(task)
        shared_context = context or {}
        while not policy.is_terminal(state):
            if state.step_count >= task.interaction.max_steps:
                raise StepLimitExceeded(
                    f"task {task.id} exceeded {task.interaction.max_steps} steps"
                )
            actor_ids = policy.next_actors(state)
            if not actor_ids:
                raise RuntimeError("policy is non-terminal but selected no actors")
            missing = [actor for actor in actor_ids if actor not in participants]
            if missing:
                raise KeyError(f"policy selected missing participants: {missing}")
            actions = await asyncio.gather(*[
                participants[actor].act(
                    policy.observe(state, actor), context=shared_context
                )
                for actor in actor_ids
            ])
            state = policy.transition(state, list(actions))
        return policy.finalize(state)


class DirectResponseEnvironment(InteractionEnvironment):
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
