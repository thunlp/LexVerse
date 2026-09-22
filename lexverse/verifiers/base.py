from __future__ import annotations

from typing import Protocol

from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask
from lexverse.verifiers.result import VerifierResult


class TaskVerifier(Protocol):
    name: str

    async def verify(
        self,
        task: LexVerseTask,
        result: EnvironmentResult,
        *,
        context: dict | None = None,
    ) -> VerifierResult: ...

