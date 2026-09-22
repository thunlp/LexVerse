from __future__ import annotations

import asyncio

from lexverse.interaction.participants.base import Participant
from lexverse.interaction.policies.base import InteractionPolicy
from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask


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
