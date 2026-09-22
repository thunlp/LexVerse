from __future__ import annotations

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.runtime.upstream import LEXEVAL
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from lexverse.tasks.selection import TaskSelection

from .loader import LexEvalLoader


class LexEvalImporter(TaskImporter):
    def __init__(self, *, offline: bool = True, catalog: dict):
        self.offline = offline
        self.catalog = catalog["tasks"]

    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]:
        loader = LexEvalLoader(offline=self.offline)
        tasks: list[LexVerseTask] = []
        for task_type in selection.select_types(list(self.catalog)):
            samples = loader.load(task_type)
            if selection.sample_ids is not None:
                wanted = set(selection.sample_ids)
                samples = [s for s in samples if str(s.index) in wanted]
            if selection.limit_per_task is not None:
                samples = samples[:selection.limit_per_task]
            spec = self.catalog[task_type]
            for sample in samples:
                tasks.append(LexVerseTask(
                    id=f"lexeval:{task_type}:{sample.index}",
                    source=SourceRef(name="lexeval", version=LEXEVAL.commit,
                                     task_type=task_type, sample_id=str(sample.index)),
                    input=TaskInput(instruction=sample.instruction, content=sample.input_text),
                    participants=[ParticipantSpec(id="assistant", role="respondent")],
                    interaction=InteractionSpec(policy="direct_response", max_steps=1),
                    evaluation=EvaluationSpec(verifier="lexeval", metrics=[spec["metric"]],
                                              reference=sample.answer,
                                              config={"task_type": task_type}),
                    raw_payload={"input": sample.input_text, "answer": sample.answer},
                ))
        return tasks
