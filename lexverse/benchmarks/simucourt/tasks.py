"""Six-category SimuCourt case importer with isolated derived references."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from lexverse.benchmarks.plugin import TaskImporter
from lexverse.benchmarks.catalog import load_catalog
from lexverse.tasks.schema import (
    EvaluationSpec,
    InteractionSpec,
    LexVerseTask,
    ParticipantSpec,
    SourceRef,
    TaskInput,
)
from . import UPSTREAM

PATCHES = Path(__file__).parent / "patches"
PROTOCOL = "simucourt_single_candidate_v1"

KINDS = {"刑事": "criminal", "民事": "civil", "行政": "administrative"}
FILES = {"first": "SimuCourt(1).json", "second": "SimuCourt(2).json"}
COMMON = ("案号", "类别", "案由", "审理程序")
FIRST = {
    "plaintiff": ("原告（公诉）", "原告诉请判令（公诉机关指控）", "原告陈述"),
    "defendant": ("被告", "被告基本情况", "被告陈述", "被告代理人辩护"),
}
SECOND = {
    "plaintiff": ("上诉人", "上诉人基本情况", "上诉人辩护", "上诉请求"),
    "defendant": ("被上诉人", "被上诉人辩护"),
}


def source_inputs(settings=None, *, offline=False):
    settings = settings or {}
    if settings.get("data_dir"):
        root = Path(settings["data_dir"]).expanduser().resolve()
    else:
        root = UPSTREAM.ensure(offline=offline) / "data"
    paths = {name: root / name for name in FILES.values()}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"SimuCourt dataset missing: {name} in {root}")
    return paths


def present(value):
    return isinstance(value, str) and value.strip() not in {"", "无"}


def derived_reference(row, stage):
    category = KINDS[row["类别"]]
    laws = [
        row[f"引用法律条文{i}"] for i in range(1, 12)
        if present(row.get(f"引用法律条文{i}"))
    ]
    if category == "criminal":
        judgment = {
            key: row[f"刑事_{key}"] for key in ("罪名", "刑期", "罚金")
            if isinstance(row.get(f"刑事_{key}"), str)
        }
    else:
        judgment = {
            f"结果{i}": row[f'{row["类别"]}结果{i}'] for i in range(1, 4)
            if present(row.get(f'{row["类别"]}结果{i}'))
        }
    reference = {
        "category": category,
        "opinion": row.get("法院意见" if stage == "first" else "二审意见", ""),
        "laws": laws,
        "judgment": judgment,
    }
    if stage == "second":
        reference["first_judgment"] = row.get("一审法院审判结果", "")
    return reference


def prompt_templates(path):
    """Read literal native templates without executing upstream top-level I/O."""
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"FIRST_TEMPLATE", "SECOND_TEMPLATE"}:
                    values[target.id] = ast.literal_eval(node.value)
    return values


def candidate_prompt(row, stage, templates):
    values = {"case_type": row["类别"] + "案件"}
    if stage == "first":
        fields = {
            "case_plaintiff": "原告（公诉）",
            "case_defendant": "被告",
            "case_defendant_detail": "被告基本情况",
            "first_fact": "基本案情",
            "claim": "原告诉请判令（公诉机关指控）",
            "plaintiff_statement": "原告陈述",
            "defendant_statement": "被告陈述",
            "defense": "被告代理人辩护",
        }
    else:
        fields = {
            "case_plaintiff": "上诉人",
            "case_defendant": "被上诉人",
            "case_defendant_detail": "上诉人基本情况",
            "first_fact": "一审法院认定事实",
            "first_analysis": "一审法院意见",
            "first_laws": "一审法院审判依据",
            "first_result": "一审法院审判结果",
            "second_appeal": "上诉请求",
            "second_fact": "本院二审查明事实",
        }
    values.update({key: row.get(field, "无") for key, field in fields.items()})
    return templates[f"{stage.upper()}_TEMPLATE"].format(**values)


class SimuCourtImporter(TaskImporter):
    def __init__(self, settings=None, *, offline=False):
        self.settings = settings or {}
        self.catalog = load_catalog(Path(__file__).with_name("catalog.yaml"))
        self.paths = source_inputs(self.settings, offline=offline)
        self.root = UPSTREAM.ensure_patched(PATCHES, offline=offline)
        self.templates = prompt_templates(self.root / "AgentsCourt/single_candidate_prompt.py")
        self.systems = {
            stage: (self.root / "AgentsCourt/system_prompt" / name).read_text(encoding="utf-8")
            for stage, name in (
                ("first", "judge_analysis_record.txt"),
                ("second", "judge_analysis_record2.txt"),
            )
        }

    def import_tasks(self, selection):
        grouped = {f"{stage}_{kind}": [] for stage in FILES for kind in KINDS.values()}
        for stage, name in FILES.items():
            rows = json.loads(self.paths[name].read_text(encoding="utf-8"))
            for row in rows:
                sid = row["案号"].strip()
                category = KINDS[row["类别"]]
                kind = f"{stage}_{category}"
                roles = FIRST if stage == "first" else SECOND
                fact_key = "基本案情" if stage == "first" else "一审法院认定事实"
                common = {key: row[key] for key in COMMON if key in row}
                common["facts"] = row[fact_key]
                if stage == "second":
                    for key in (
                        "一审法院认定事实", "一审法院意见",
                        "一审法院审判依据", "一审法院审判结果", "本院二审查明事实",
                    ):
                        if key in row:
                            common[key] = row[key]
                materials = {
                    role: {key: row[key] for key in fields if key in row}
                    for role, fields in roles.items()
                }
                user_prompt = candidate_prompt(row, stage, self.templates)
                grouped[kind].append(LexVerseTask(
                    id=f"simucourt:{kind}:{sid}",
                    source=SourceRef(
                        name="simucourt", version=UPSTREAM.commit,
                        task_type=kind, sample_id=sid,
                    ),
                    input=TaskInput(
                        instruction=user_prompt,
                        messages=[
                            {"role": "system", "content": self.systems[stage]},
                            {"role": "user", "content": user_prompt},
                        ],
                    ),
                    participants=[ParticipantSpec(id="assistant", role="assistant")],
                    interaction=InteractionSpec(policy="direct_response", max_steps=1),
                    evaluation=EvaluationSpec(
                        verifier="simucourt",
                        metrics=self.catalog["tasks"][kind]["metrics"],
                        reference=derived_reference(row, stage),
                    ),
                    metadata={"stage": stage, "category": category, "protocol": PROTOCOL},
                    raw_payload={"common": common, "materials": materials},
                ))
        tasks = []
        for kind in selection.select_types(list(grouped)):
            tasks.extend(selection.apply(
                kind, grouped[kind], sample_id=lambda task: task.source.sample_id,
            ))
        return tasks
