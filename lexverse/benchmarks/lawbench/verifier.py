from __future__ import annotations

import tempfile
from pathlib import Path

from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask
from lexverse.verifiers.result import VerifierProvenance, VerifierResult

from .evaluate import LawBenchEvaluator, prediction_to_disk


class LawBenchTaskVerifier:
    name = "lawbench"

    def __init__(self, *, offline: bool = True):
        self.offline = offline
        self._evaluator = None

    async def verify(self, task: LexVerseTask, result: EnvironmentResult, *, context=None):
        if result.answer is None:
            return VerifierResult(status="invalid", provenance=self._provenance())
        evaluator = self._evaluator or LawBenchEvaluator(offline=self.offline)
        self._evaluator = evaluator
        task_type = task.source.task_type or ""
        with tempfile.TemporaryDirectory(prefix="lexverse-lawbench-") as tmp:
            path = prediction_to_disk(
                Path(tmp) / f"{task_type}.json",
                [task.raw_payload.get("origin_prompt", task.input.prompt)],
                [result.answer],
                [str(task.evaluation.reference)],
            )
            score = evaluator.score_file(task_type, path)
        return VerifierResult(
            status="completed",
            metrics={"score": score.score, "abstention_rate": score.abstention_rate},
            raw_result={"task_type": task_type},
            provenance=self._provenance(),
        )

    def verify_run(
        self,
        cases: list[tuple[LexVerseTask, str]],
        *,
        predictions_root: Path,
        model_name: str,
    ):
        """Write upstream prediction files and score each complete task once."""
        evaluator = self._evaluator or LawBenchEvaluator(offline=self.offline)
        self._evaluator = evaluator
        model_dir = predictions_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[tuple[LexVerseTask, str]]] = {}
        for task, response in cases:
            grouped.setdefault(task.source.task_type or "", []).append((task, response))
        scores = []
        for task_type, task_cases in grouped.items():
            path = prediction_to_disk(
                model_dir / f"{task_type}.json",
                [task.raw_payload.get("origin_prompt", task.input.prompt)
                 for task, _ in task_cases],
                [response for _, response in task_cases],
                [str(task.evaluation.reference) for task, _ in task_cases],
            )
            scores.append(evaluator.score_file(task_type, path))
        return scores

    def _provenance(self):
        from lexverse.runtime.upstream import LAWBENCH
        return VerifierProvenance(name=self.name, version="1", upstream_commit=LAWBENCH.commit)
