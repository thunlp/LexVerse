"""Select public case/persona cells without constructing gold-bearing prompts."""
from __future__ import annotations

import json
from pathlib import Path

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import dataset_files, load_catalog
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from . import UPSTREAM

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
PERSONAS = list(load_catalog(CATALOG_PATH)["tasks"])


def load_cases(root):
    cases = []
    for path in dataset_files(load_catalog(CATALOG_PATH), root=root).values():
        for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not text.strip():
                continue
            row = json.loads(text)
            sid = row.get("case_id") or row.get("id")
            if not sid or not row.get("facts") or not row.get("opening"):
                raise ValueError(f"invalid native case: {path.name}:{line}")
            cases.append((sid, row, path.name, line))
    if len({cell[0] for cell in cases}) != len(cases):
        raise ValueError("duplicate native case IDs")
    return cases


def source_inputs(root):
    paths = list(dataset_files(load_catalog(CATALOG_PATH), root=root).values())
    paths += [path for path in (root / "config").rglob("*") if path.is_file()]
    return {str(path.relative_to(root)): path for path in sorted(paths)}


class DLawBenchImporter(TaskImporter):
    def __init__(self, *, offline=False, upstream=UPSTREAM):
        self.root = upstream.ensure(offline=offline)
        self.version = upstream.commit
        self.catalog = load_catalog(CATALOG_PATH)
        self.cases = load_cases(self.root)
        self.personas = json.loads((self.root / "config/personas.json").read_text())
        if set(PERSONAS) != {key for key in self.personas if not key.startswith("_")}:
            raise ValueError("native persona catalog changed")

    def import_tasks(self, selection):
        tasks = []
        for persona in selection.select_types(PERSONAS):
            for sid, row, filename, line in selection.apply(
                persona, self.cases, sample_id=lambda cell: cell[0],
            ):
                tasks.append(LexVerseTask(
                    id=f"dlawbench:{persona}:{sid}",
                    source=SourceRef(name="dlawbench", version=self.version, task_type=persona,
                                     sample_id=sid, provenance={"jurisdiction": row["jurisdiction"]}),
                    input=TaskInput(instruction=row["opening"]),
                    participants=[ParticipantSpec(id="lawyer", role="lawyer"),
                                  ParticipantSpec(id="client", role="client")],
                    interaction=InteractionSpec(policy="dlawbench_native"),
                    evaluation=EvaluationSpec(verifier="dlawbench", metrics=list(self.catalog["tasks"][persona]["metrics"])),
                    metadata={"persona_key": persona, "jurisdiction": row["jurisdiction"],
                              "case_file": filename, "case_line": line},
                    raw_payload=row,
                ))
        return tasks
