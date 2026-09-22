from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask


class ExecutionEnvironment(ABC):
    name: str

    async def prepare(self, context: dict | None = None) -> None:
        return None

    @abstractmethod
    async def run(self, task: LexVerseTask, work_dir: Path) -> EnvironmentResult:
        pass

    async def close(self) -> None:
        return None
