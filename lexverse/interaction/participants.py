"""Participant interface and model-backed participant."""
from __future__ import annotations

from typing import Protocol
from lexverse.interaction.schema import ParticipantAction
from dataclasses import dataclass, field
from lexverse.providers.base import Provider


class Participant(Protocol):
    id: str

    async def act(self, observation: list[dict], *, context: dict) -> ParticipantAction: ...


@dataclass
class ModelParticipant:
    id: str
    provider: Provider
    generation_config: dict = field(default_factory=dict)

    async def act(self, observation: list[dict], *, context: dict) -> ParticipantAction:
        response = await self.provider.generate(observation, config=self.generation_config)
        return ParticipantAction(
            actor_id=self.id,
            content=response.content,
            metadata={"model": response.model, "usage": response.usage},
        )


__all__ = ["ModelParticipant", "Participant"]
