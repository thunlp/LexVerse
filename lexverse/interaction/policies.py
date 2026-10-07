"""Interaction rules and their registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from lexverse.interaction.schema import EnvironmentResult, ParticipantAction, InteractionState
from lexverse.tasks.schema import LexVerseTask
from collections.abc import Callable


class InteractionPolicy(ABC):
    @abstractmethod
    def initialize(self, task: LexVerseTask) -> InteractionState: ...

    @abstractmethod
    def is_terminal(self, state: InteractionState) -> bool: ...

    @abstractmethod
    def next_actors(self, state: InteractionState) -> list[str]: ...

    @abstractmethod
    def observe(self, state: InteractionState, actor_id: str) -> list[dict]: ...

    @abstractmethod
    def transition(self, state: InteractionState, actions: list[ParticipantAction]) -> InteractionState: ...

    @abstractmethod
    def finalize(self, state: InteractionState) -> EnvironmentResult: ...


class DirectResponsePolicy(InteractionPolicy):
    def initialize(self, task: LexVerseTask) -> InteractionState:
        messages = task.input.messages or [
            {"role": "user", "content": task.input.prompt}
        ]
        return InteractionState(messages=messages, data={"actor": task.participants[0].id})

    def is_terminal(self, state: InteractionState) -> bool:
        return state.answer is not None

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

    def finalize(self, state: InteractionState) -> EnvironmentResult:
        return EnvironmentResult(
            status="completed",
            answer=state.answer,
            messages=state.messages,
            dialog_history=state.dialog_history,
            final_state=state.model_dump(mode="json"),
        )


PolicyFactory = Callable[[], InteractionPolicy]
_POLICIES: dict[str, PolicyFactory] = {"direct_response": DirectResponsePolicy}


def register_policy(name: str, factory: PolicyFactory) -> None:
    if name in _POLICIES:
        raise ValueError(f"interaction policy {name!r} is already registered")
    _POLICIES[name] = factory


def create_policy(name: str) -> InteractionPolicy:
    try:
        return _POLICIES[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_POLICIES)) or "<none>"
        raise KeyError(f"unknown interaction policy {name!r}; available: {available}") from exc


def available_policies() -> list[str]:
    return sorted(_POLICIES)


__all__ = [
    "DirectResponsePolicy", "InteractionPolicy", "available_policies",
    "create_policy", "register_policy",
]
