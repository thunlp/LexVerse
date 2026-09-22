from __future__ import annotations

from dataclasses import dataclass, field

from lexverse.interaction.result import ParticipantAction
from lexverse.providers.base import Provider


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

