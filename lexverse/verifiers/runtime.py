from __future__ import annotations

import asyncio
from pathlib import Path

from lexverse.interaction.result import EnvironmentResult
from lexverse.runtime.artifacts import atomic_write_json
from lexverse.tasks.schema import LexVerseTask

from .base import TaskVerifier
from .result import VerifierResult


class VerifierRuntime:
    async def run(
        self,
        verifier: TaskVerifier,
        task: LexVerseTask,
        result: EnvironmentResult,
        output_dir: Path,
        *,
        timeout_sec: float = 600.0,
        context: dict | None = None,
    ) -> VerifierResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        verified = await asyncio.wait_for(
            verifier.verify(task, result, context=context), timeout=timeout_sec
        )
        atomic_write_json(output_dir / "result.json", verified.model_dump(mode="json"))
        return verified
