from __future__ import annotations

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.runtime.upstream import J1BENCH
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from lexverse.tasks.selection import TaskSelection

from .loader import J1BenchLoader
from .scenarios import require_supported


class J1BenchImporter(TaskImporter):
    def __init__(
        self,
        *,
        scenarios: list[str] | tuple[str, ...] | None = None,
        case_databases: dict[str, object] | None = None,
        catalog: dict,
        # Backward-compatible single-scenario construction for Python callers.
        scenario: str | None = None,
        case_database=None,
    ):
        self.scenarios = tuple(scenarios or ([scenario] if scenario else ["CI"]))
        self.case_databases = dict(case_databases or {})
        if scenario and case_database is not None:
            self.case_databases[scenario] = case_database
        self.catalog = catalog["tasks"]

    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]:
        tasks: list[LexVerseTask] = []
        wanted = set(selection.sample_ids) if selection.sample_ids is not None else None
        for scenario_name in self.scenarios:
            scenario = require_supported(scenario_name)
            loader = J1BenchLoader(
                self.case_databases.get(scenario_name), scenario=scenario_name
            )
            cases = loader.load_all()
            if wanted is not None:
                cases = [case for case in cases if case.id in wanted]
            if selection.limit_per_task is not None:
                cases = cases[:selection.limit_per_task]
            for case in cases:
                tasks.append(LexVerseTask(
                    id=f"j1bench:{scenario_name}:{case.id}",
                    source=SourceRef(
                        name="j1bench", version=J1BENCH.commit,
                        task_type=scenario_name, sample_id=case.id,
                    ),
                    input=TaskInput(
                        instruction=f"运行 J1Bench {scenario_name} 场景案件 {case.id}"
                    ),
                    participants=[
                        ParticipantSpec(id=role, role=role) for role in scenario.roles
                    ],
                    interaction=InteractionSpec(
                        policy=f"j1bench_{scenario_name.lower()}", max_steps=50,
                        policy_config={"scenario": scenario_name},
                    ),
                    evaluation=EvaluationSpec(
                        verifier="j1bench", execution="subprocess",
                        metrics=list(self.catalog[scenario_name]["metrics"]),
                        config={"scenario": scenario_name},
                    ),
                    raw_payload=case.raw,
                ))
        return tasks
