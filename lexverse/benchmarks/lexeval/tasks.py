from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import dataset_files, load_catalog
from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.sources import UpstreamSource
from lexverse.benchmarks.lexeval import UPSTREAM as LEXEVAL
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from lexverse.tasks.bundle import TaskSelection


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
        self._files = dataset_files(load_catalog(Path(__file__).with_name("catalog.yaml")), root=self._cache)
        self._gen_dir = self._cache / "code" / "generation"
        if not (self._gen_dir / "model_gen.py").exists():
            raise EvaluationError(
                f"upstream missing code/generation/model_gen.py at {self._gen_dir}"
            )
        if str(self._gen_dir) not in sys.path:
            sys.path.insert(0, str(self._gen_dir))
        self._model_gen = importlib.import_module("model_gen")

    def data_dir(self) -> Path:
        return next(iter(self._files.values())).parent

    def task_file(self, task_id: str) -> Path:
        matches = [path for path in self._files.values() if path.name == f"{task_id}.json"]
        if len(matches) != 1:
            raise EvaluationError(f"LexEval catalog must declare one {task_id}.json")
        return matches[0]

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


class LexEvalImporter(TaskImporter):
    def __init__(self, *, offline: bool = True, catalog: dict,
                 few_shot_path: Path | None = None):
        self.offline = offline
        self.catalog = catalog["tasks"]
        self.few_shot_path = few_shot_path

    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]:
        loader = LexEvalLoader(offline=self.offline)
        tasks: list[LexVerseTask] = []
        for task_type in selection.select_types(list(self.catalog)):
            samples = loader.load(task_type, few_shot_path=self.few_shot_path)
            samples = selection.apply(task_type, samples, sample_id=lambda sample: sample.index)
            spec = self.catalog[task_type]
            for sample in samples:
                tasks.append(LexVerseTask(
                    id=f"lexeval:{task_type}:{sample.index}",
                    source=SourceRef(name="lexeval", version=LEXEVAL.commit,
                                     task_type=task_type, sample_id=str(sample.index)),
                    input=TaskInput(instruction=sample.instruction, content=sample.input_text),
                    participants=[ParticipantSpec(id="assistant", role="respondent")],
                    interaction=InteractionSpec(policy="direct_response", max_steps=1),
                    evaluation=EvaluationSpec(verifier="lexeval", metrics=list(spec["metrics"]),
                                              reference=sample.answer,
                                              config={"task_type": task_type}),
                    raw_payload={"input": sample.input_text, "answer": sample.answer},
                ))
        return tasks
