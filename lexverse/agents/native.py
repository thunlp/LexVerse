"""Benchmark-owned execution lifecycle; world rules stay in each benchmark."""
from abc import ABC, abstractmethod
from pathlib import Path
from lexverse.models.trial.result import EnvironmentResult
from lexverse.models.task.task import LexVerseTask


class NativeRunner(ABC):
    async def prepare(self, context: dict | None = None) -> None:
        pass

    @abstractmethod
    async def run(self, task: LexVerseTask, work_dir: Path) -> EnvironmentResult: ...

    async def close(self) -> None:
        pass

