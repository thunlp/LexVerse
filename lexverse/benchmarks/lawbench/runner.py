"""Generate LawBench predictions in its native on-disk format.

Files use ``<prediction_dir>/<model_name>/<task_id>.json`` so the upstream
evaluator can consume them unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lexverse.benchmarks.lawbench.evaluate import prediction_to_disk
from lexverse.benchmarks.lawbench.loader import LawBenchLoader, LawBenchSample, Shot
from lexverse.providers.base import Provider


@dataclass
class RunSummary:
    task_id: str
    shot: Shot
    model_name: str
    samples: int
    unsuccessful: int
    output_path: Path


class LawBenchRunner:
    def __init__(
        self,
        loader: LawBenchLoader,
        provider: Provider,
        model_name: str,
    ):
        self._loader = loader
        self._provider = provider
        self._model_name = model_name

    async def run(
        self,
        task_id: str,
        output_dir: Path,
        *,
        shot: Shot = "zero_shot",
        limit: int | None = None,
    ) -> RunSummary:
        samples = self._loader.load(task_id, shot=shot)
        if limit is not None:
            samples = samples[:limit]

        origin_prompts: list[str] = []
        predictions: list[str] = []
        refrs: list[str] = []
        unsuccessful = 0
        for s in samples:
            response = await self._provider.generate(
                [{"role": "user", "content": s.prompt}]
            )
            out = response.content
            if out == "未成功回答":
                unsuccessful += 1
            origin_prompts.append(s.prompt)
            predictions.append(out)
            refrs.append(s.answer)

        # Upstream layout: <pred_dir>/<model>/<task>.json
        target_dir = output_dir / self._model_name
        target = prediction_to_disk(
            target_dir / f"{task_id}.json",
            origin_prompts, predictions, refrs,
        )
        return RunSummary(
            task_id=task_id,
            shot=shot,
            model_name=self._model_name,
            samples=len(samples),
            unsuccessful=unsuccessful,
            output_path=target,
        )
