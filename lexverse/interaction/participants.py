from __future__ import annotations

from typing import Protocol
from pathlib import Path
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


@dataclass
class AgentParticipant:
    id: str
    config: object

    async def act(self, observation: list[dict], *, context: dict) -> ParticipantAction:
        from lexverse.runtime.enhancement import act
        result = await act(self.config, observation, context)
        return ParticipantAction(actor_id=self.id, content=result["answer"], metadata={
            "capabilities": True, "trace_ref": result["trace_ref"], "usage": result["usage"],
            "result_ref": str(Path(result["run_dir"]) / "result.json"), "resource_hash": result["resource_hash"],
        })


__all__ = ["ModelParticipant", "AgentParticipant", "Participant"]
