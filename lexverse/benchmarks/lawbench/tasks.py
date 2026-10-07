from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import dataset_files, load_catalog
from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.sources import UpstreamSource
from lexverse.benchmarks.lawbench import UPSTREAM as LAWBENCH
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from lexverse.tasks.bundle import TaskSelection


Shot = Literal["zero_shot", "one_shot"]


# LawBench task family metadata (used for coverage matrix + defaults).
# The 20 tasks upstream ships. `2-1` (wsjd) requires an external Chinese
# grammar tool (parallel_to_m2 + subprocess); we surface but do not gate it.
ALL_TASK_IDS: tuple[str, ...] = (
    "1-1", "1-2",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "2-6", "2-7", "2-8", "2-9", "2-10",
    "3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8",
)


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
        self._files = dataset_files(load_catalog(Path(__file__).with_name("catalog.yaml")), root=self._cache)

    def task_file(self, task_id: str, shot: Shot) -> Path:
        matches = [path for path in self._files.values() if path.name == f"{task_id}.json" and path.parent.name == shot]
        if len(matches) != 1:
            raise EvaluationError(f"LawBench catalog must declare one {shot}/{task_id}.json")
        return matches[0]

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


class LawBenchImporter(TaskImporter):
    def __init__(self, *, offline: bool = True, shot: str = "zero_shot", catalog: dict):
        self.offline = offline
        self.shot = shot
        self.catalog = catalog["tasks"]

    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]:
        loader = LawBenchLoader(offline=self.offline)
        tasks: list[LexVerseTask] = []
        for task_type in selection.select_types(list(self.catalog)):
            samples = loader.load(task_type, shot=self.shot)
            samples = selection.apply(task_type, samples, sample_id=lambda sample: sample.index)
            for sample in samples:
                tasks.append(LexVerseTask(
                    id=f"lawbench:{task_type}:{self.shot}:{sample.index}",
                    source=SourceRef(name="lawbench", version=LAWBENCH.commit,
                                     task_type=task_type, sample_id=str(sample.index)),
                    input=TaskInput(instruction=sample.instruction + "\n", content=sample.question),
                    participants=[ParticipantSpec(id="assistant", role="respondent")],
                    interaction=InteractionSpec(policy="direct_response", max_steps=1),
                    evaluation=EvaluationSpec(verifier="lawbench",
                                              metrics=list(self.catalog[task_type]["metrics"]),
                                              reference=sample.answer,
                                              config={"task_type": task_type, "shot": self.shot}),
                    raw_payload={"origin_prompt": sample.prompt, "answer": sample.answer},
                ))
        return tasks
