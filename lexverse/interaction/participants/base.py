from __future__ import annotations

from typing import Protocol

from lexverse.interaction.result import ParticipantAction


class Participant(Protocol):
    id: str

    async def act(self, observation: list[dict], *, context: dict) -> ParticipantAction: ...

