"""Load LawBench's native zero-shot and one-shot JSON files."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.upstream import LAWBENCH, UpstreamSource


Shot = Literal["zero_shot", "one_shot"]


@dataclass(frozen=True)
class LawBenchSample:
    task_id: str       # "1-1", "3-2", ...
    shot: Shot
    index: int
    instruction: str
    question: str
    answer: str

    @property
    def prompt(self) -> str:
        # Upstream evaluation main.py assumes the model was prompted with
        # instruction + question (no per-task suffix like LexEval).
        # `predictions/*/*.json` shows `origin_prompt = [{role: HUMAN,
        # prompt: instruction + '\n' + question}]`. We match that exactly.
        return f"{self.instruction}\n{self.question}"


class LawBenchLoader:
    def __init__(self, upstream: UpstreamSource = LAWBENCH, offline: bool = False):
        self._upstream = upstream
        self._cache = upstream.ensure_patched(
            Path(__file__).parent / "patches", offline=offline
        )
        self._data_dir = self._cache / "data"
        if not self._data_dir.exists():
            raise EvaluationError(f"upstream missing data/ at {self._data_dir}")

    def task_file(self, task_id: str, shot: Shot) -> Path:
        p = self._data_dir / shot / f"{task_id}.json"
        if not p.exists():
            raise EvaluationError(f"LawBench task file not found: {p}")
        return p

    def load(self, task_id: str, shot: Shot = "zero_shot") -> list[LawBenchSample]:
        raw = json.loads(self.task_file(task_id, shot).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise EvaluationError(
                f"expected JSON array in {task_id}.json, got {type(raw).__name__}"
            )
        return [
            LawBenchSample(
                task_id=task_id,
                shot=shot,
                index=i,
                instruction=item["instruction"],
                question=item["question"],
                answer=item["answer"],
            )
            for i, item in enumerate(raw)
        ]


# LawBench task family metadata (used for coverage matrix + defaults).
# The 20 tasks upstream ships. `2-1` (wsjd) requires an external Chinese
# grammar tool (parallel_to_m2 + subprocess); we surface but do not gate it.
ALL_TASK_IDS: tuple[str, ...] = (
    "1-1", "1-2",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "2-6", "2-7", "2-8", "2-9", "2-10",
    "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8",
)
