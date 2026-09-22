from __future__ import annotations

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.runtime.upstream import LAWBENCH
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from lexverse.tasks.selection import TaskSelection

from .loader import LawBenchLoader


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
            if selection.sample_ids is not None:
                wanted = set(selection.sample_ids)
                samples = [s for s in samples if str(s.index) in wanted]
            if selection.limit_per_task is not None:
                samples = samples[:selection.limit_per_task]
            for sample in samples:
                tasks.append(LexVerseTask(
                    id=f"lawbench:{task_type}:{self.shot}:{sample.index}",
                    source=SourceRef(name="lawbench", version=LAWBENCH.commit,
                                     task_type=task_type, sample_id=str(sample.index)),
                    input=TaskInput(instruction=sample.instruction + "\n", content=sample.question),
                    participants=[ParticipantSpec(id="assistant", role="respondent")],
                    interaction=InteractionSpec(policy="direct_response", max_steps=1),
                    evaluation=EvaluationSpec(verifier="lawbench",
                                              metrics=["score", "abstention_rate"],
                                              reference=sample.answer,
                                              config={"task_type": task_type, "shot": self.shot}),
                    raw_payload={"origin_prompt": sample.prompt, "answer": sample.answer},
                ))
        return tasks
