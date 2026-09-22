from __future__ import annotations

from abc import ABC, abstractmethod

from lexverse.interaction.result import EnvironmentResult, ParticipantAction
from lexverse.interaction.state import InteractionState
from lexverse.tasks.schema import LexVerseTask


class InteractionPolicy(ABC):
    @abstractmethod
    def initialize(self, task: LexVerseTask) -> InteractionState: ...

    @abstractmethod
    def next_actors(self, state: InteractionState) -> list[str]: ...

    @abstractmethod
    def observe(self, state: InteractionState, actor_id: str) -> list[dict]: ...

    @abstractmethod
    def transition(self, state: InteractionState, actions: list[ParticipantAction]) -> InteractionState: ...

    @abstractmethod
    def is_terminal(self, state: InteractionState) -> bool: ...

    @abstractmethod
    def finalize(self, state: InteractionState) -> EnvironmentResult: ...

