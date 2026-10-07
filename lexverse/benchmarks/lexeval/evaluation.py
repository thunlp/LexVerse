from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Literal

from lexverse.verifiers.base import (
    TaskVerifier, BenchmarkAggregator,
    mean_metrics,
    RunEvaluationResult,
    VerifierProvenance,
)
from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.sources import UpstreamSource
from lexverse.runtime.results import artifact_model_name
from lexverse.benchmarks.lexeval import UPSTREAM as LEXEVAL
from lexverse.tasks.schema import LexVerseTask


_FILENAME_RE = re.compile(r"(?P<i>[1-6])_(?P<j>[1-6])\.jsonl$")

Metric = Literal["Accuracy", "F1", "Rouge_L", "Bertscore", "Bartscore"]
TaskType = Literal["multiple_choice", "generation"]


def _rouge_warning() -> str | None:
    """Reject a Chinese scorer exposed through LexEval's plain rouge import."""
    try:
        from rouge import Rouge
    except ImportError:
        return None  # The native evaluator reports a missing scoring dependency.
    provides = getattr(Rouge, "__module__", "")
    if "rouge_chinese" in provides:
        return (
            "rouge_chinese shadows plain rouge: `from rouge import Rouge` "
            f"resolves to {provides!r}. LexEval Rouge_L requires plain rouge. "
            "Use separate virtualenvs, or restore plain rouge after installing rouge_chinese: "
            "python -m pip install --force-reinstall --no-deps 'rouge==1.0.1'. "
            "LawBench uses rouge_chinese directly."
        )
    return None


@dataclass(frozen=True)
class LexEvalScore:
    file: Path
    task_id: str          # "i_j"
    task_type: TaskType
    metric: Metric
    score: float


def parse_task_id_from_filename(path: Path) -> str:
    match = _FILENAME_RE.search(path.name)
    if match is None:
        raise EvaluationError(
            f"cannot parse LexEval task id from filename: {path.name} "
            "(expected suffix like '_1_1.jsonl')"
        )
    return f"{match.group('i')}_{match.group('j')}"


def task_type_for(task_id: str) -> TaskType:
    """LexEval convention: family 5 (i=5) is generation, everything else is MC.

    See upstream evaluate.py::main.
    """
    if not re.fullmatch(r"[1-6]_[1-6]", task_id):
        raise EvaluationError(f"invalid LexEval task id: {task_id!r} (expected 'i_j', 1<=i,j<=6)")
    return "generation" if task_id.startswith("5_") else "multiple_choice"


def write_scores_csv(scores: Iterable[LexEvalScore], out_path: Path, *, model_name: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        # Upstream pandas.to_csv keeps its index column.
        writer.writerow(["", "task", "model", "metrics", "score"])
        for index, score in enumerate(scores):
            writer.writerow([
                index, score.task_id, model_name, score.metric, score.score
            ])
    return out_path


class LexEvalEvaluator:
    """Load the evaluator in an isolated import context."""

    def __init__(self, upstream: UpstreamSource = LEXEVAL, offline: bool = False):
        self._upstream = upstream
        patches_dir = Path(__file__).parent / "patches"
        cache = upstream.ensure_patched(patches_dir, offline=offline)
        eval_dir = cache / "code" / "evaluation"
        if not (eval_dir / "evaluate.py").exists():
            raise EvaluationError(
                f"upstream missing code/evaluation/evaluate.py at {eval_dir}"
            )
        self._eval_dir = eval_dir
        self._install_sys_path()
        module = import_module("evaluate")
        self._Evaluator = getattr(module, "Evaluator")

    def score_file(
        self,
        jsonl_path: Path,
        metric: Metric,
        *,
        task_id: str | None = None,
        device: str = "cpu",
        model_path: str | None = None,
    ) -> LexEvalScore:
        if task_id is None:
            task_id = parse_task_id_from_filename(jsonl_path)
        ttype = task_type_for(task_id)

        if ttype == "multiple_choice" and metric not in {"Accuracy", "F1"}:
            raise EvaluationError(
                f"multiple_choice tasks accept Accuracy/F1, not {metric!r}"
            )
        if ttype == "generation" and metric not in {"Rouge_L", "Bertscore", "Bartscore"}:
            raise EvaluationError(
                f"generation tasks accept Rouge_L/Bertscore/Bartscore, not {metric!r}"
            )

        if metric == "Rouge_L":
            conflict = _rouge_warning()
            if conflict:
                raise EvaluationError(conflict)

        try:
            evaluator = self._Evaluator(
                file_path=str(jsonl_path),
                task_type=ttype,
                metric=metric,
                device=device,
                model_path=model_path,
            )
            raw = evaluator.eval()
        except Exception as exc:  # upstream raises ValueError, IOError, etc.
            raise EvaluationError(
                f"upstream Evaluator failed on {jsonl_path.name} "
                f"(task={task_id}, metric={metric}): {type(exc).__name__}: {exc}"
            ) from exc

        return LexEvalScore(
            file=jsonl_path,
            task_id=task_id,
            task_type=ttype,
            metric=metric,
            score=float(raw),
        )

    def score_run(
        self,
        native_output_dir: Path,
        *,
        choice_metric: Metric = "Accuracy",
        generation_metric: Metric = "Rouge_L",
        device: str = "cpu",
        model_path: str | None = None,
    ) -> list[LexEvalScore]:
        """Score every *.jsonl under a run directory (non-recursive).

        Follows upstream evaluate.py::main: family 5 uses generation_metric,
        everything else uses choice_metric.
        """
        if not native_output_dir.is_dir():
            raise EvaluationError(f"not a directory: {native_output_dir}")

        scores: list[LexEvalScore] = []
        for path in sorted(native_output_dir.glob("*.jsonl")):
            task_id = parse_task_id_from_filename(path)
            metric = generation_metric if task_type_for(task_id) == "generation" else choice_metric
            scores.append(
                self.score_file(
                    path,
                    metric=metric,
                    task_id=task_id,
                    device=device,
                    model_path=model_path,
                )
            )
        return scores

    def _install_sys_path(self) -> None:
        p = str(self._eval_dir)
        if p not in sys.path:
            sys.path.insert(0, p)


class LexEvalTaskVerifier(TaskVerifier):
    name = "lexeval"
    grouped = True

    def __init__(self, *, offline: bool = True):
        self.offline = offline
        self._evaluator = None

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        cases = [
            (task, records[task.id]["response"]) for task in tasks
            if records.get(task.id, {}).get("status") == "completed"
            and isinstance(records[task.id].get("response"), str)
        ]
        scores = self.verify_run(
            cases, predictions_root=run_dir / "verifier" / "predictions",
            model_name=model_name,
        ) if cases else []
        return RunEvaluationResult(expected_task_ids=[task.id for task in tasks], groups=[{
            "task_type": score.task_id,
            "granularity": "task_type",
            "artifacts": {"predictions": str(
                Path("verifier") / "predictions" / artifact_model_name(model_name) / f"{artifact_model_name(model_name)}_{score.task_id}.jsonl"
            )},
            "task_ids": [task.id for task, _ in cases if task.source.task_type == score.task_id],
            "expected_task_ids": [task.id for task in tasks if task.source.task_type == score.task_id],
            "metrics": {score.metric: score.score},
            "provenance": self._provenance().model_dump(mode="json"),
        } for score in scores])

    def write_results(self, result, tasks, records, *, run_dir, model_name):
        scores = [LexEvalScore(Path(group["artifacts"]["predictions"]), group["task_type"], "",
                               metric, score) for group in result.groups
                  for metric, score in group["metrics"].items()]
        write_scores_csv(scores, run_dir / "evaluation_result.csv", model_name=model_name)

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
        model_dir = predictions_root / artifact_model_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[tuple[LexVerseTask, str]]] = {}
        for task, response in cases:
            grouped.setdefault(task.source.task_type or "", []).append((task, response))
        scores = []
        for task_type, task_cases in grouped.items():
            path = model_dir / f"{artifact_model_name(model_name)}_{task_type}.jsonl"
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
        from lexverse.benchmarks.lexeval import UPSTREAM as LEXEVAL
        return VerifierProvenance(name=self.name, version="1.0", upstream_commit=LEXEVAL.commit)


class LexEvalAggregator(BenchmarkAggregator):
    def aggregate(self, records: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
        tasks = mean_metrics(records)
        return {"benchmark": "lexeval", "model": model_name, "tasks": tasks}

    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            # Match upstream LexEval's public result columns.
            writer.writerow(["task", "model", "metrics", "score"])
            for task, result in summary["tasks"].items():
                for metric, score in result["metrics"].items():
                    writer.writerow([task, model_name, metric, f"{score:.6f}"])
        return path
