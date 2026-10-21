"""Import MILE evaluation cases while keeping reference outputs isolated to the verifier."""
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

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
TASK_TYPE = "complaint_generation"
PROFILE_KEYS = (
    "case_r_f", "plaintiff_type", "plaintiff", "law_level", "personality",
    "speak_style", "less", "defendant", "evidence", "claim",
)
FILES = {"gold": "evaluate_data/evaluate_data.json"}
PRIOR = {"policy": "none", "reason": None, "laws": None}


def max_turns(config):
    turns = (config or {}).get("max_turns", 15)
    if type(turns) is not int or not 1 <= turns <= 15:
        raise ValueError("MILE generation.max_turns must be an integer in 1—15")
    return turns


def data_root(config=None, *, upstream=UPSTREAM):
    config = config or {}
    if config.get("data_dir"):
        root = Path(config["data_dir"]).expanduser().resolve()
        for candidate in (root, root / "evaluation", root / "src/evaluation"):
            if (candidate / FILES["gold"]).is_file():
                return candidate
        raise ValueError("MILE data_dir missing evaluate_data/evaluate_data.json")
    return upstream.ensure(offline=config.get("offline", False)) / "src/evaluation"


def source_inputs(config=None):
    root = data_root(config)
    paths = {name: root / filename for name, filename in FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"MILE data missing: {path}")
    return paths


def parse_literal(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def build_profile(row):
    profile = {key: row[key] for key in PROFILE_KEYS}
    for key, fields in (
        ("claim", ("非诉讼费用内容", "诉讼费用内容")),
        ("speak_style", ("tone", "content", "interaction")),
    ):
        value = parse_literal(row[key])
        profile[key] = {field: value[field] for field in fields}
    evidence = parse_literal(row["evidence"])
    profile["evidence"] = {"原告": list(evidence["原告"])}
    return profile


def validate_history(value):
    history = parse_literal(value)
    if not isinstance(history, list) or not history:
        raise ValueError("MILE dialog_history must be a nonempty list")
    if history[-1].get("role") != "Lawyer" or not str(history[-1].get("content", "")).strip():
        raise ValueError("MILE dialog_history missing final complaint")
    return history


class MILEImporter(TaskImporter):
    def __init__(self, config=None, *, upstream=UPSTREAM):
        self.config = dict(config or {})
        self.turns = max_turns(self.config)
        self.version = upstream.commit
        self.catalog = load_catalog(CATALOG_PATH)
        root = data_root(self.config, upstream=upstream)
        rows = json.loads((root / FILES["gold"]).read_text(encoding="utf-8"))
        self.gold = {str(row["id"]): row for row in rows}
        self.profiles = {str(row["id"]): build_profile(row) for row in rows}

    def import_tasks(self, selection):
        tasks = []
        roles = ("lawyer", "client", "supervisor")
        for kind in selection.select_types([TASK_TYPE]):
            selected = selection.apply(
                kind, list(self.gold.values()), sample_id=lambda row: row["id"],
            )
            for gold in selected:
                sid = str(gold["id"])
                tasks.append(LexVerseTask(
                    id=f"maser:{TASK_TYPE}:{sid}",
                    source=SourceRef(
                        name="maser", version=self.version,
                        task_type=kind, sample_id=sid,
                    ),
                    input=TaskInput(instruction="您好，我想写一份起诉状。"),
                    participants=[ParticipantSpec(id=role, role=role) for role in roles],
                    interaction=InteractionSpec(
                        policy="mile_native",
                        max_steps=self.turns,
                    ),
                    evaluation=EvaluationSpec(
                        verifier="maser",
                        reference=gold,
                        metrics=list(self.catalog["tasks"][kind]["metrics"]),
                    ),
                    raw_payload={"profile": self.profiles[sid], "prior": dict(PRIOR)},
                ))
        return tasks


MASERImporter = MILEImporter
