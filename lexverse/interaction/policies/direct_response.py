from __future__ import annotations

from lexverse.interaction.result import EnvironmentResult, ParticipantAction
from lexverse.interaction.state import InteractionState
from lexverse.tasks.schema import LexVerseTask

from .base import InteractionPolicy


class DirectResponsePolicy(InteractionPolicy):
    def initialize(self, task: LexVerseTask) -> InteractionState:
        messages = task.input.messages or [
            {"role": "user", "content": task.input.prompt}
        ]
        return InteractionState(messages=messages, data={"actor": task.participants[0].id})

    def next_actors(self, state: InteractionState) -> list[str]:
        return [] if state.answer is not None else [state.data["actor"]]

    def observe(self, state: InteractionState, actor_id: str) -> list[dict]:
        return list(state.messages)

    def transition(self, state: InteractionState, actions: list[ParticipantAction]) -> InteractionState:
        if len(actions) != 1:
            raise ValueError("direct_response requires exactly one action")
        action = actions[0]
        state.answer = action.content
        state.messages.append({"role": "assistant", "content": action.content})
        state.dialog_history.append({"actor": action.actor_id, "content": action.content})
        state.step_count += 1
        return state

    def is_terminal(self, state: InteractionState) -> bool:
        return state.answer is not None

    def finalize(self, state: InteractionState) -> EnvironmentResult:
        return EnvironmentResult(
            status="completed",
            answer=state.answer,
            messages=state.messages,
            dialog_history=state.dialog_history,
            final_state=state.model_dump(mode="json"),
        )
