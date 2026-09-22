"""Load LexEval samples through the upstream prompt builder."""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.upstream import LEXEVAL, UpstreamSource


@dataclass(frozen=True)
class LexEvalSample:
    """One evaluation item: prompt materials + gold answer.

    `instruction` and `input_text` come straight from upstream
    `process_prompt`, including few-shot examples and task-specific answer,
    summary, judgment-analysis, and translation-result suffixes.
    `answer` is upstream's `answer` field; scoring compares model output
    to it after the same normalization the upstream Evaluator uses.
    """

    task_id: str
    index: int
    instruction: str
    input_text: str
    answer: str

    @property
    def prompt(self) -> str:
        return self.instruction + self.input_text


class LexEvalLoader:
    """Preserve upstream prompt assembly, including few-shot suffixes."""

    def __init__(self, upstream: UpstreamSource = LEXEVAL, offline: bool = False):
        self._upstream = upstream
        self._cache = upstream.ensure_patched(
            Path(__file__).parent / "patches", offline=offline
        )
        self._gen_dir = self._cache / "code" / "generation"
        if not (self._gen_dir / "model_gen.py").exists():
            raise EvaluationError(
                f"upstream missing code/generation/model_gen.py at {self._gen_dir}"
            )
        if str(self._gen_dir) not in sys.path:
            sys.path.insert(0, str(self._gen_dir))
        self._model_gen = importlib.import_module("model_gen")

    def data_dir(self) -> Path:
        return self._cache / "data"

    def task_file(self, task_id: str) -> Path:
        path = self.data_dir() / f"{task_id}.json"
        if not path.exists():
            raise EvaluationError(f"LexEval task file not found: {path}")
        return path

    def load(
        self,
        task_id: str,
        *,
        few_shot_path: Path | None = None,
    ) -> list[LexEvalSample]:
        task_file = self.task_file(task_id)
        gen = self._model_gen.model_generator(
            f_path=str(task_file),
            is_few_shot=few_shot_path is not None,
            device="cpu",
            is_vllm=False,
            model_path=None,
            few_shot_path=str(few_shot_path) if few_shot_path else None,
        )
        instrs, inputs, answers = gen.process_prompt(task_id)
        return [
            LexEvalSample(
                task_id=task_id,
                index=i,
                instruction=instrs[i],
                input_text=inputs[i],
                answer=answers[i],
            )
            for i in range(len(instrs))
        ]
