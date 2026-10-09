from __future__ import annotations

import ast
import json
from itertools import zip_longest
import re
from pathlib import Path

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import dataset_files, load_catalog
from lexverse.tasks.bundle import TaskSelection
from lexverse.tasks.schema import (
    EvaluationSpec, InteractionSpec, LexVerseTask, ParticipantSpec, SourceRef, TaskInput,
)
from . import UPSTREAM

FILES = {
    "case_analysis": {None: "practical_case_analysis_250.jsonl"},
    "legal_consultation": {"mid": "public_legal_consultation_18.json"},
    "document_generation": {
        "plaintiff": "Plantiffs_Statement.json", "defendant": "Defendants_Statement.json",
    },
}


def prompt_templates(path):
    """Read literal upstream templates; never execute the generator's file I/O."""
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"prompt_template_dict", "user_prompt_dict"}:
                    values[target.id] = ast.literal_eval(node.value)
    return values["prompt_template_dict"], values["user_prompt_dict"]


def consultation_score(rubric, sid):
    declared = re.search(r"总分\s*[：:]\s*(\d+(?:\.\d+)?)", rubric)
    entries = re.findall(r"^\s*\d+\s*[.．、]\s*[（(]\s*\+\s*(\d+(?:\.\d+)?)\s*分\s*[）)]", rubric, re.M)
    lines = re.findall(r"^\s*\d+\s*[.．、]", rubric, re.M)
    if not entries or len(entries) != len(lines):
        raise ValueError(f"consultation rubric has unrecognized point entries: {sid}")
    total = sum(float(value) for value in entries)
    source = "rubric_item_sum"
    if declared and float(declared.group(1)) != total:
        source = "rubric_item_sum_conflicts_with_total"
    return str(total), source


class PLawBenchImporter(TaskImporter):
    def __init__(self, *, offline=False, upstream=UPSTREAM):
        self.root = upstream.ensure_patched(Path(__file__).parent / "patches", offline=offline)
        self.version = upstream.commit
        self.catalog = load_catalog(Path(__file__).with_name("catalog.yaml"))
        self.files = {path.name: path for path in dataset_files(self.catalog, root=self.root).values()}
        self.templates = prompt_templates(self.root / "prompt_generate.py")

    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]:
        tasks = []
        for kind in selection.select_types(list(self.catalog["tasks"])):
            batches = []
            for subtype, filename in FILES[kind].items():
                path = self.files[filename]
                rows = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                        if path.suffix == ".jsonl" else json.loads(path.read_text()))
                batch = []
                for index, row in enumerate(rows):
                    sid = str(index) if kind == "case_analysis" else str(row["id"])
                    if kind == "document_generation":
                        sid = f"{subtype}:{sid}"
                    batch.append((sid, row, subtype))
                batches.append(batch)
            # Interleave document roles so a two-case smoke includes both roles.
            samples = [sample for group in zip_longest(*batches) for sample in group if sample is not None]
            if len({sid for sid, _, _ in samples}) != len(samples):
                raise ValueError(f"duplicate PLawBench IDs in {kind}")
            for sid, row, subtype in selection.apply(kind, samples, sample_id=lambda sample: sample[0]):
                tasks.append(self._task(kind, sid, row, subtype))
        return tasks

    def _task(self, kind, sid, row, subtype=None):
        systems, users = self.templates
        if kind == "case_analysis":
            prompt = systems[kind] + users[kind].format(context=row["context"], question=row["question"])
            scorer = "case_analysis_merge"
        else:
            template = f"{kind}_{subtype}" if kind == "legal_consultation" else kind
            prompt = systems[template] + users[kind].format(question=row["conversation"])
            scorer = "legal_qa" if kind == "legal_consultation" else kind
        rubric = row.get("rubrics") or row.get("Rubrics")
        if not rubric:
            raise ValueError(f"missing PLawBench rubric: {kind}:{sid}")
        metadata = {"label": row["label"] if kind == "case_analysis" else row.get("tag")}
        if kind == "legal_consultation":
            metadata["difficulty"] = subtype
        elif kind == "document_generation":
            metadata["document_role"] = subtype
        item = dict(row, prompt=prompt, task_name=f"{kind}_{subtype}" if kind == "legal_consultation" else kind)
        score_source = "original"
        if scorer == "legal_qa" and not item.get("score"):
            item["score"], score_source = consultation_score(rubric, sid)
        if scorer == "document_generation" and not item.get("score"):
            raise ValueError(f"document score missing: {kind}:{sid}")
        metrics = list(self.catalog["tasks"][kind]["metrics"])
        return LexVerseTask(
            id=f"plawbench:{kind}:{sid}",
            source=SourceRef(name="plawbench", version=self.version, task_type=kind, sample_id=sid,
                             provenance={key: metadata[key] for key in ("difficulty", "document_role") if key in metadata}),
            input=TaskInput(instruction=prompt),
            participants=[ParticipantSpec(id="assistant", role="respondent")],
            interaction=InteractionSpec(policy="direct_response", max_steps=1),
            evaluation=EvaluationSpec(verifier="plawbench", metrics=metrics, reference=rubric,
                                      config={"task_type": scorer, "item": item}),
            metadata={**metadata, "score_source": score_source},
            raw_payload=row,
        )
