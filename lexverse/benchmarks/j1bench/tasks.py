from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import dataset_files, load_catalog
from lexverse.runtime.errors import TrialError
from lexverse.benchmarks.j1bench import UPSTREAM as J1BENCH
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from lexverse.tasks.bundle import TaskSelection

from .scenarios import require_supported


CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
DATASET_URL = f"https://huggingface.co/datasets/{load_catalog(CATALOG_PATH)['dataset']['repo_id']}"
J1_EVAL_DATASET_URL = DATASET_URL + "/tree/" + load_catalog(CATALOG_PATH)["dataset"]["revision"]


@dataclass(frozen=True)
class J1BenchCase:
    """One case record from a J1-Eval scenario file (whole JSON kept opaque).

    Different scenarios have different top-level keys; we only require
    `id` for identification and `raw` for round-trip serialization.
    """

    scenario: str
    id: str
    raw: dict[str, Any] = field(repr=False)


def resolve_case_database(scenario: str, *, offline: bool = False) -> Path:
    """Prepare all data files at the catalog revision, then select one scenario."""
    files = dataset_files(load_catalog(CATALOG_PATH), offline=offline)
    filename = f"J1-Eval_{scenario}.jsonl"
    if filename not in files:
        raise TrialError(f"J1Bench dataset is missing {filename}")
    return files[filename]


class J1BenchLoader:
    def __init__(self, case_database: Path | None = None, scenario: str = "CI") -> None:
        self.scenario = scenario
        if case_database is None:
            raise TrialError(
                "J1Bench requires benchmark.case_database to point to an explicitly "
                f"downloaded J1-Eval_{scenario}.jsonl; source: {J1_EVAL_DATASET_URL}"
            )
        self.case_database = Path(case_database)

    def load_all(self) -> list[J1BenchCase]:
        if not self.case_database.exists():
            raise TrialError(
                f"J1Bench case database not found: {self.case_database}"
            )
        cases: list[J1BenchCase] = []
        with self.case_database.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TrialError(
                        f"invalid JSON at {self.case_database}:{line_no}: {exc}"
                    ) from exc
                if "id" not in obj:
                    raise TrialError(
                        f"case missing 'id' at {self.case_database}:{line_no}"
                    )
                cases.append(J1BenchCase(
                    scenario=self.scenario, id=str(obj["id"]), raw=obj
                ))
        return cases

    def by_id(self, case_id: str) -> J1BenchCase:
        for case in self.load_all():
            if case.id == case_id:
                return case
        raise TrialError(
            f"case {case_id!r} not found in {self.case_database}"
        )


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
        for scenario_name in selection.select_types(self.scenarios):
            scenario = require_supported(scenario_name)
            loader = J1BenchLoader(
                self.case_databases.get(scenario_name), scenario=scenario_name
            )
            cases = loader.load_all()
            cases = selection.apply(scenario_name, cases, sample_id=lambda case: case.id)
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
