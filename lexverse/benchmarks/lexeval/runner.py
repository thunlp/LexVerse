"""Generate LexEval's native JSONL prediction files.

The upstream evaluator derives model and task identifiers from filenames in
the form ``<model_name>_<i>_<j>.jsonl``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lexverse.benchmarks.lexeval.loader import LexEvalLoader, LexEvalSample
from lexverse.providers.base import Provider
from lexverse.runtime.artifacts import atomic_write_text


@dataclass
class RunSummary:
    task_id: str
    model_name: str
    samples: int
    unsuccessful: int
    output_path: Path


class LexEvalRunner:
    def __init__(
        self,
        loader: LexEvalLoader,
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
        few_shot_path: Path | None = None,
        limit: int | None = None,
    ) -> RunSummary:
        """Generate predictions for `task_id`; return a summary.

        `limit`: if set, only run the first N samples (smoke tests).
        `few_shot_path`: forwarded to loader; upstream expects
            `<task>_few_shot.json` for few-shot runs.
        """
        samples = self._loader.load(task_id, few_shot_path=few_shot_path)
        if limit is not None:
            samples = samples[:limit]

        lines: list[str] = []
        unsuccessful = 0
        for sample in samples:
            response = await self._provider.generate(
                [{"role": "user", "content": sample.prompt}]
            )
            output = response.content
            record = {
                "input": sample.input_text,
                "output": output,
                "answer": sample.answer,
            }
            if output == "未成功回答":
                unsuccessful += 1
            lines.append(json.dumps(record, ensure_ascii=False))

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self._model_name}_{task_id}.jsonl"
        # Newline-terminated JSONL (upstream jsonlines reader handles either)
        atomic_write_text(output_path, "\n".join(lines) + "\n")

        return RunSummary(
            task_id=task_id,
            model_name=self._model_name,
            samples=len(samples),
            unsuccessful=unsuccessful,
            output_path=output_path,
        )
