from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
from lexverse.tasks.bundle import TaskBundle

if TYPE_CHECKING:
    from lexverse.benchmarks.plugin import BenchmarkPlugin


@dataclass
class CompletenessReport:
    expected: int
    present: int
    missing: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing and not self.duplicated

    def to_dict(self) -> dict:
        return {
            "expected": self.expected,
            "present": self.present,
            "missing": self.missing,
            "duplicated": self.duplicated,
            "is_complete": self.is_complete,
        }


def safe_path_component(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in str(value)
    )


def _safe_id(value: str) -> str:
    return safe_path_component(value)


def artifact_model_name(name: str) -> str:
    """Keep HF IDs and absolute weight paths inside the run's artifact tree."""
    import re
    import hashlib
    if re.fullmatch(r"[\w.-]+", name) and name not in {".", ".."}:
        return name
    readable = re.sub(r"[^\w.-]+", "-", Path(name).name).strip(".-") or "model"
    return readable + "-" + hashlib.sha256(name.encode()).hexdigest()[:8]


def trial_relative_path(task_type: str, sample_id: str) -> Path:
    return Path(safe_path_component(task_type)) / safe_path_component(sample_id)


def trial_result_paths(trials_root: Path) -> list[Path]:
    """List canonical Trial results plus legacy singular filenames."""
    paths = {
        *trials_root.glob("*/results.json"),
        *trials_root.glob("*/*/results.json"),
        *trials_root.glob("*/result.json"),
        *trials_root.glob("*/*/result.json"),
    }
    return sorted(path for path in paths if path.parent.name != "verifier")


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via temp file + rename so partial writes never appear."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def check_completeness(
    expected_ids: list[str],
    present_ids: list[str],
) -> CompletenessReport:
    expected_set = set(expected_ids)
    present_set = set(present_ids)
    missing = sorted(expected_set - present_set)
    seen: dict[str, int] = {}
    for pid in present_ids:
        seen[pid] = seen.get(pid, 0) + 1
    duplicated = sorted(pid for pid, count in seen.items() if count > 1)
    return CompletenessReport(
        expected=len(expected_set),
        present=len(present_set & expected_set),
        missing=missing,
        duplicated=duplicated,
    )


def _migrate_legacy_trial(
    trial_dir: Path, *, task: Any, benchmark: str, model_name: str
) -> dict[str, Any]:
    """Merge the former trial/environment/verifier triplet into results.json."""
    state_path = trial_dir / "trial.json"
    environment_path = trial_dir / "environment-result.json"
    state = _read_json(state_path)
    environment = _read_json(environment_path)
    verifier = _read_json(trial_dir / "verifier" / "result.json")
    if not state and not environment and not verifier:
        return {}
    status = (
        "completed" if verifier.get("status") == "completed"
        else state.get("status", "failed")
    )
    record = {
        "schema_version": 1.0,
        "id": task.id,
        "benchmark": benchmark,
        "task": task.source.task_type or benchmark,
        "sample_id": task.source.sample_id,
        "trial_id": state.get("trial_id", _safe_id(task.id)),
        "attempt": state.get("attempt", 1),
        "status": status,
        "config_hash": state.get("config_hash", ""),
        "work_dir": str(trial_dir),
        "input": task.input.model_dump(mode="json"),
        "reference": task.evaluation.reference,
        "response": environment.get("answer"),
        "interaction": {
            "messages": environment.get("messages") or [],
            "dialog_history": environment.get("dialog_history") or [],
            "final_state": environment.get("final_state") or {},
            "trace": environment.get("trace") or [],
        },
        "model": {"name": model_name},
        "evaluation": verifier or {"status": "pending", "metrics": {}},
        "artifacts": {
            **(environment.get("artifacts") or {}),
            **(verifier.get("artifacts") or {}),
        },
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "error": state.get("error"),
    }
    atomic_write_json(trial_dir / "results.json", record)
    state_path.unlink(missing_ok=True)
    environment_path.unlink(missing_ok=True)
    return record


def _collect_trial_results(
    run_dir: Path, bundle: TaskBundle, *, model_name: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in bundle.tasks:
        canonical_relative = trial_relative_path(
            task.source.task_type or bundle.benchmark,
            task.source.sample_id,
        )
        canonical_dir = run_dir / "trials" / canonical_relative
        legacy_dir = run_dir / "trials" / _safe_id(task.id)
        trial_dir = canonical_dir if canonical_dir.exists() else legacy_dir
        path = trial_dir / "results.json"
        if not path.exists():
            path = trial_dir / "result.json"
        record = _read_json(path)
        if not record:
            record = _migrate_legacy_trial(
                trial_dir, task=task, benchmark=bundle.benchmark,
                model_name=model_name,
            )
        if record:
            record.setdefault("id", task.id)
            record.setdefault("benchmark", bundle.benchmark)
            record.setdefault("task", task.source.task_type or bundle.benchmark)
            record.setdefault("sample_id", task.source.sample_id)
        if not record:
            record = {
                "schema_version": 1.0,
                "id": task.id,
                "benchmark": bundle.benchmark,
                "task": task.source.task_type or bundle.benchmark,
                "sample_id": task.source.sample_id,
                "status": "missing",
                "evaluation": {"status": "pending", "metrics": {}},
                "error": None,
            }
        records.append(record)
    return records


def _read_all_trial_results(run_dir: Path) -> list[dict[str, Any]]:
    records = [
        value for path in trial_result_paths(run_dir / "trials")
        if (value := _read_json(path))
    ]
    return records


def _write_compact_manifest(
    run_dir: Path, manifest: dict[str, Any], bundle: TaskBundle, *, model_name: str
) -> None:
    if manifest.get("snapshot_version") in (1, 2):
        atomic_write_json(run_dir / "manifest.json", {
            key: value for key, value in manifest.items() if key != "bundle"
        })
        return
    compact = {
        "schema_version": manifest.get("schema_version", 1.0),
        "created_at": manifest.get("created_at"),
        "config_hash": manifest.get("config_hash"),
        "config_source": manifest.get("config_source"),
        "architecture_version": manifest.get("architecture_version"),
        "provenance": manifest.get("provenance"),
        "run_hash": manifest.get("run_hash"),
        "benchmark": bundle.benchmark,
        "source_version": bundle.source_version,
        "model": model_name,
        "task_types": sorted({
            task.source.task_type for task in bundle.tasks if task.source.task_type
        }),
        "sample_count": len(bundle.tasks),
        "prepared_bundle_hash": bundle.content_hash(),
    }
    atomic_write_json(run_dir / "manifest.json", compact)


def finalize_run(
    run_dir: Path,
    manifest: dict[str, Any],
    plugin: BenchmarkPlugin,
    *,
    model_name: str,
    task_metrics: dict[str, dict[str, float]] | None = None,
    write_evaluation_csv: bool = True,
) -> dict[str, Any]:
    bundle = (
        TaskBundle.model_validate(manifest["bundle"])
        if "bundle" in manifest else None
    )
    records = (
        _collect_trial_results(run_dir, bundle, model_name=model_name)
        if bundle is not None else _read_all_trial_results(run_dir)
    )
    expected = len(bundle.tasks) if bundle is not None else int(
        manifest.get("sample_count", len(records))
    )
    completed = sum(record.get("status") == "completed" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    missing = [record["id"] for record in records if record.get("status") == "missing"]
    complete = completed == expected and not failed and not missing
    evaluation = _read_json(run_dir / "evaluation.json")
    if (evaluation and evaluation.get("status") != "completed") or (
        manifest.get("provenance") and not evaluation
    ):
        complete = False

    aggregator = plugin.create_aggregator()
    aggregate = aggregator.aggregate(records, model_name=model_name)
    if task_metrics is not None:
        for task, metrics in task_metrics.items():
            if task in aggregate["tasks"]:
                aggregate["tasks"][task]["metrics"] = metrics
    job_result = {
        "schema_version": 1.0,
        "benchmark": bundle.benchmark if bundle is not None else manifest["benchmark"],
        "model": model_name,
        "status": "completed" if complete else "incomplete",
        "samples": {
            "expected": expected,
            "completed": completed,
            "failed": failed,
            "missing": missing,
        },
        "tasks": aggregate["tasks"],
    }
    if plugin.name == "dlawbench":
        job_result.update(overall=aggregate["overall"], by_jurisdiction=aggregate["by_jurisdiction"])
    if "coverage" in manifest:
        job_result["coverage"] = manifest["coverage"]
    if evaluation:
        job_result["evaluation"] = {
            key: evaluation[key] for key in (
                "status", "expected_task_ids", "scored_task_ids", "missing_task_ids"
            ) if key in evaluation
        }
    elif manifest.get("provenance"):
        job_result["evaluation"] = {"status": "pending"}
    atomic_write_json(run_dir / "summary.json", job_result)
    if write_evaluation_csv:
        aggregator.write_csv(
            aggregate, run_dir / "evaluation_result.csv", model_name=model_name
        )
    if complete:
        if bundle is not None:
            _write_compact_manifest(run_dir, manifest, bundle, model_name=model_name)
    return job_result
