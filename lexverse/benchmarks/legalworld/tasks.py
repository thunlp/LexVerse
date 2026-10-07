"""Import case/party/range cells from the pinned public dataset."""
from __future__ import annotations

import json
from pathlib import Path

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import dataset_files, load_catalog
from lexverse.tasks.schema import EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput
from . import UPSTREAM

STAGES = ["LC", "DRAFT", "CI", "SD", "APPEAL_DRAFT", "CIA"]
CATALOG_PATH = Path(__file__).with_name("catalog.yaml")


def stage_range(start, end):
    start, end = str(start).upper(), str(end).upper()
    if start not in STAGES or end not in STAGES or STAGES.index(start) > STAGES.index(end):
        raise ValueError("Legal-world requires valid start_stage <= end_stage in native stage order")
    return STAGES[STAGES.index(start):STAGES.index(end) + 1]


def task_ranges(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Legal-world requires benchmark.tasks with id, party_role, start_stage and end_stage")
    ranges = {}
    identities = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Legal-world tasks must define id, party_role, start_stage and end_stage")
        name, role = entry.get("id"), entry.get("party_role")
        if not isinstance(name, str) or not name:
            raise ValueError("Legal-world task id must be a non-empty string")
        if role not in {"plaintiff", "defendant"}:
            raise ValueError("Legal-world party_role must be plaintiff or defendant")
        stages = stage_range(entry.get("start_stage"), entry.get("end_stage"))
        identity = (role, stages[0], stages[-1])
        if name in ranges or identity in identities:
            raise ValueError("duplicate Legal-world task id or case/party/range")
        ranges[name] = identity
        identities.add(identity)
    return ranges


def ensure_dataset(*, offline=False):
    files = dataset_files(load_catalog(CATALOG_PATH), offline=offline)
    return files["light_case_dataset.json"]


def source_inputs(*, offline=True):
    root = UPSTREAM.ensure_patched(Path(__file__).with_name("patches"), offline=offline)
    paths = [path for folder in ("backend/legal-skillhub/public", "backend/src/prompts", "backend/src/pipeline")
             for path in (root / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    return {**{f"dataset:{name}": path for name, path in dataset_files(load_catalog(CATALOG_PATH), offline=offline).items()},
            **{str(path.relative_to(root)): path for path in sorted(paths)}}


class LegalWorldImporter(TaskImporter):
    def __init__(self, *, offline=False, task_definitions=None):
        self.ranges = task_ranges(task_definitions) if task_definitions is not None else {}
        self.catalog = load_catalog(CATALOG_PATH)
        self.root = UPSTREAM.ensure(offline=offline)
        self.path = ensure_dataset(offline=offline)
        self.cases = json.loads(self.path.read_text(encoding="utf-8"))
        ids = [case.get("original_id") for case in self.cases]
        if any(type(sid) is not int for sid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Legal-world requires unique integer original_id values")
        for case in self.cases:
            parties = case.get("extracted_info", {}).get("party_info", {})
            if not parties.get("plaintiff") or not parties.get("defendant"):
                raise ValueError(f"missing party materials: original_id={case['original_id']}")

    def import_tasks(self, selection):
        if not self.ranges:
            raise ValueError("Legal-world requires explicit task definitions")
        tasks = []
        for kind in selection.select_types(list(self.ranges)):
            role, start, end = self.ranges[kind]
            for case in selection.apply(kind, self.cases, sample_id=lambda row: str(row["original_id"])):
                sid = str(case["original_id"])
                tasks.append(LexVerseTask(
                    id=f"legalworld:{kind}:{sid}",
                    source=SourceRef(name="legalworld", version=UPSTREAM.commit, task_type=kind,
                                     sample_id=sid, provenance={"dataset_revision": self.catalog["dataset"]["revision"]}),
                    input=TaskInput(instruction=f"Run native {role} pipeline from {start} through {end}."),
                    participants=[ParticipantSpec(id="lawyer", role="lawyer"),
                                  ParticipantSpec(id="simulation", role="simulation")],
                    interaction=InteractionSpec(policy="legalworld_native"),
                    evaluation=EvaluationSpec(verifier="legalworld"),
                    metadata={"original_id": case["original_id"], "party_role": role,
                              "start_stage": start, "end_stage": end},
                    # Gold stays in the evaluator dataset, never in the public task prompt/payload.
                    raw_payload={"original_id": case["original_id"]},
                ))
        return tasks
