"""Call LexEval's upstream evaluator without reimplementing its metrics.

The evaluator directory must be placed on ``sys.path`` because its modules
import each other as top-level names rather than as a package.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal

from lexverse.compatibility import rouge_warning
from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.upstream import LEXEVAL, UpstreamSource

_FILENAME_RE = re.compile(r"(?P<i>[1-6])_(?P<j>[1-6])\.jsonl$")

Metric = Literal["Accuracy", "F1", "Rouge_L", "Bertscore", "Bartscore"]
TaskType = Literal["multiple_choice", "generation"]


@dataclass(frozen=True)
class LexEvalScore:
    file: Path
    task_id: str          # "i_j"
    task_type: TaskType
    metric: Metric
    score: float


def task_type_for(task_id: str) -> TaskType:
    """LexEval convention: family 5 (i=5) is generation, everything else is MC.

    See upstream evaluate.py::main.
    """
    if not re.fullmatch(r"[1-6]_[1-6]", task_id):
        raise EvaluationError(f"invalid LexEval task id: {task_id!r} (expected 'i_j', 1<=i,j<=6)")
    return "generation" if task_id.startswith("5_") else "multiple_choice"


def parse_task_id_from_filename(path: Path) -> str:
    match = _FILENAME_RE.search(path.name)
    if match is None:
        raise EvaluationError(
            f"cannot parse LexEval task id from filename: {path.name} "
            "(expected suffix like '_1_1.jsonl')"
        )
    return f"{match.group('i')}_{match.group('j')}"


class LexEvalEvaluator:
    """Load the evaluator in an isolated import context."""

    def __init__(self, upstream: UpstreamSource = LEXEVAL, offline: bool = False):
        self._upstream = upstream
        # Surface a rouge/rouge_chinese conflict once, rather than silently
        # producing drifty Rouge_L numbers (see lexverse/compatibility.py).
        _warn = rouge_warning()
        if _warn:
            print(f"[lexverse] {_warn}", file=sys.stderr)
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

    def _install_sys_path(self) -> None:
        p = str(self._eval_dir)
        if p not in sys.path:
            sys.path.insert(0, p)

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
