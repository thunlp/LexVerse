from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask
from lexverse.verifiers.result import VerifierProvenance, VerifierResult

from .evaluate import LexEvalEvaluator


class LexEvalTaskVerifier:
    name = "lexeval"

    def __init__(self, *, offline: bool = True):
        self.offline = offline
        self._evaluator = None

    async def verify(self, task: LexVerseTask, result: EnvironmentResult, *, context=None):
        if result.answer is None:
            return VerifierResult(status="invalid", provenance=self._provenance())
        evaluator = self._evaluator or LexEvalEvaluator(offline=self.offline)
        self._evaluator = evaluator
        task_type = task.source.task_type or ""
        metric = task.evaluation.metrics[0]
        with tempfile.TemporaryDirectory(prefix="lexverse-lexeval-") as tmp:
            path = Path(tmp) / f"model_{task_type}.jsonl"
            record = {"input": task.raw_payload.get("input", task.input.content),
                      "output": result.answer, "answer": task.evaluation.reference}
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            score = evaluator.score_file(path, metric=metric, task_id=task_type)
        return VerifierResult(status="completed", metrics={metric: score.score},
                              raw_result={"task_type": task_type}, provenance=self._provenance())

    def verify_run(
        self,
        cases: list[tuple[LexVerseTask, str]],
        *,
        predictions_root: Path,
        model_name: str,
    ):
        """Write upstream prediction files and score each complete task once."""
        evaluator = self._evaluator or LexEvalEvaluator(offline=self.offline)
        self._evaluator = evaluator
        model_dir = predictions_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[tuple[LexVerseTask, str]]] = {}
        for task, response in cases:
            grouped.setdefault(task.source.task_type or "", []).append((task, response))
        scores = []
        for task_type, task_cases in grouped.items():
            path = model_dir / f"{model_name}_{task_type}.jsonl"
            lines = [json.dumps({
                "input": task.raw_payload.get("input", task.input.content),
                "output": response,
                "answer": task.evaluation.reference,
            }, ensure_ascii=False) for task, response in task_cases]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            scores.append(evaluator.score_file(
                path, metric=task_cases[0][0].evaluation.metrics[0], task_id=task_type
            ))
        return scores

    def _provenance(self):
        from lexverse.runtime.upstream import LEXEVAL
        return VerifierProvenance(name=self.name, version="1", upstream_commit=LEXEVAL.commit)
