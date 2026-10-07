from __future__ import annotations

import math
import hashlib
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from lexverse.tasks.schema import LexVerseTask


class VerifierProvenance(BaseModel):
    name: str
    version: str
    upstream_commit: str | None = None
    config_hash: str | None = None
    model: str | None = None


class VerifierResult(BaseModel):
    status: Literal["completed", "failed", "invalid", "not_applicable"]
    metrics: dict[str, float | int] = Field(default_factory=dict)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    provenance: VerifierProvenance

    @field_validator("metrics")
    @classmethod
    def _finite_numeric_metrics(cls, metrics):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"metric {key!r} must be a finite number")
        return metrics


@dataclass
class RunEvaluationResult:
    """Native scoring groups and optional independently scored trial results."""

    groups: list[dict[str, Any]] = field(default_factory=list)
    trials: dict[str, VerifierResult] = field(default_factory=dict)
    expected_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        scored = {task_id for group in self.groups for task_id in group["task_ids"]}
        scored -= {task_id for task_id, result in self.trials.items() if result.status != "completed"}
        missing = sorted(set(self.expected_task_ids) - scored)
        return {
            "status": "completed" if not missing else "incomplete",
            "expected_task_ids": self.expected_task_ids,
            "scored_task_ids": sorted(scored),
            "missing_task_ids": missing,
            "groups": self.groups,
            "trials": {key: value.model_dump(mode="json") for key, value in self.trials.items()},
        }

    def task_metrics(self) -> dict[str, dict[str, float]]:
        return {group["task_type"]: group["metrics"] for group in self.groups}

    @classmethod
    def from_dict(cls, data):
        for group in data["groups"]:
            if (not isinstance(group["task_type"], str) or not isinstance(group["task_ids"], list)
                    or not all(isinstance(key, str) for key in group["task_ids"])
                    or not isinstance(group["metrics"], dict)
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in group["metrics"].values())):
                raise ValueError("invalid native scoring group")
        return cls(groups=data["groups"], expected_task_ids=data["expected_task_ids"],
                   trials={key: VerifierResult.model_validate(value) for key, value in data["trials"].items()})


def mean_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        task = str(record["task"])
        task_counts = counts.setdefault(
            task, {"expected": 0, "completed": 0, "failed": 0, "missing": 0}
        )
        task_counts["expected"] += 1
        status = str(record.get("status", "missing"))
        if status == "completed" and record.get("evaluation"):
            status = record["evaluation"].get("status", status)
            status = {"pending": "missing", "running": "missing", "invalid": "failed",
                      "not_applicable": "missing"}.get(status, status)
        if status in task_counts:
            task_counts[status] += 1
        if status != "completed":
            grouped.setdefault(task, {})
            continue
        metrics = (record.get("evaluation") or {}).get("metrics") or {}
        target = grouped.setdefault(task, {})
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                target.setdefault(name, []).append(float(value))

    return {
        task: {
            "expected_samples": counts[task]["expected"],
            "completed_samples": counts[task]["completed"],
            "failed_samples": counts[task]["failed"],
            "missing_samples": counts[task]["missing"],
            "sample_count": counts[task]["completed"],
            "metrics": {
                name: sum(values) / len(values)
                for name, values in sorted(metrics.items())
                if values
            },
        }
        for task, metrics in sorted(grouped.items())
    }


class TaskVerifier(ABC):
    """Native scoring adapters sharing one run-level recovery loop."""

    name: str
    grouped = False

    def evaluation_units(self, tasks: list[LexVerseTask]) -> dict[str, list[LexVerseTask]]:
        units = {}
        for task in tasks:
            key = task.source.task_type if self.grouped else task.id
            units.setdefault(key, []).append(task)
        return units

    def scoring_identity(self) -> dict:
        module = importlib.import_module(type(self).__module__)
        directory = Path(module.__file__).parent
        upstream = getattr(importlib.import_module(module.__package__), "UPSTREAM", None)
        root = getattr(self, "root", upstream.pristine_dir if upstream else None)
        files = sorted(directory.rglob("*.py")) + sorted(directory.rglob("*.patch"))
        if root is not None:
            files += sorted(Path(root).rglob("*.py"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(directory) if path.is_relative_to(directory)
                            else path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return {"name": self.name, "upstream_commit": upstream.commit if upstream else None,
                "implementation_sha256": digest.hexdigest()}

    async def evaluate_run(
        self, tasks: list[LexVerseTask], records: dict[str, dict], *,
        run_dir: Path, model_name: str, context: dict | None = None,
    ) -> RunEvaluationResult:
        from lexverse.runtime.orchestrator import Orchestrator
        context = dict(context or {})
        concurrency = context.pop("concurrency", 1)
        on_progress = context.pop("on_progress", None)
        result = await Orchestrator(n_concurrent=concurrency).evaluate(
            tasks, records, verifier=self, run_dir=run_dir, model_name=model_name,
            context=context, judge_config=context.get("judge_config", {}), on_progress=on_progress,
        )
        self.write_results(result, tasks, records, run_dir=run_dir, model_name=model_name)
        return result

    @abstractmethod
    async def evaluate_unit(
        self, tasks: list[LexVerseTask], records: dict[str, dict], *,
        run_dir: Path, model_name: str, context: dict,
    ) -> RunEvaluationResult: ...

    def prepare_unit(self, tasks, records, *, run_dir, model_name, context):
        """Adapters with native intermediate caches validate them before scoring."""

    def build_result(self, tasks, results):
        trials = {key: value for result in results.values() for key, value in result.trials.items()}
        groups = ([group for result in results.values() for group in result.groups]
                  if self.grouped else self.build_groups(tasks, trials))
        return RunEvaluationResult(groups=groups, trials=trials,
                                   expected_task_ids=[task.id for task in tasks])

    def build_groups(self, tasks, results):
        groups = []
        for kind in dict.fromkeys(task.source.task_type for task in tasks):
            ids = [task.id for task in tasks if task.source.task_type == kind
                   and task.id in results and results[task.id].status == "completed"]
            if ids:
                common = set.intersection(*(set(results[sid].metrics) for sid in ids))
                groups.append({"task_type": kind, "task_ids": ids, "metrics": {
                    key: sum(results[sid].metrics[key] for sid in ids) / len(ids)
                    for key in sorted(common)}})
        return groups

    def write_results(self, result, tasks, records, *, run_dir, model_name):
        """Adapters with native batch reports write them after recovery and merging."""


class BenchmarkAggregator(ABC):
    @abstractmethod
    def aggregate(
        self, records: list[dict[str, Any]], *, model_name: str
    ) -> dict[str, Any]: ...

    @abstractmethod
    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path: ...
