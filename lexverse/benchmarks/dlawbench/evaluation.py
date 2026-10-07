"""Preserve native scores and report incomplete judge coverage explicitly."""
from __future__ import annotations

import csv
import importlib.util
import json
import logging
from pathlib import Path

from lexverse.runtime.results import atomic_write_json, trial_relative_path
from lexverse.verifiers.base import TaskVerifier, BenchmarkAggregator, RunEvaluationResult, VerifierProvenance, VerifierResult, mean_metrics
from . import UPSTREAM
from .environment import connection_spec, run_bridge

logger = logging.getLogger("lexverse.run")


def load_public_metrics(root):
    spec = importlib.util.spec_from_file_location("lexverse_dlawbench_metrics", root / "scripts/metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DLawBenchTaskVerifier(TaskVerifier):
    name = "dlawbench"

    def __init__(self, *, offline=False):
        self.root = UPSTREAM.ensure(offline=offline)
        self.native_metrics = load_public_metrics(self.root)

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        context = context or {}
        connections = context["judge_connections"]
        evaluated = RunEvaluationResult(expected_task_ids=[task.id for task in tasks])
        for index, task in enumerate(tasks):
            record = records.get(task.id, {})
            if record.get("status") != "completed":
                continue
            trial_dir = run_dir / "trials" / trial_relative_path(task.source.task_type, task.source.sample_id)
            session_path = Path(record.get("execution_artifacts", record.get("artifacts", {})).get("session", trial_dir / "session.json"))
            if not session_path.is_absolute():
                session_path = run_dir / session_path
            attempts = []
            grading = None
            session = json.loads(session_path.read_text()) if session_path.is_file() else None
            if session and session.get("case_id") == (task.raw_payload.get("case_id") or task.raw_payload["id"]):
                settings = {"case": task.raw_payload, "session": session, "mode": context["mode"],
                            "panel": list(connections), "connections": {
                                key: connection_spec(value) for key, value in connections.items()}}
                for attempt in range(1, context["max_attempts"] + 1):
                    work = trial_dir / "evaluation" / f"attempt-{attempt}"
                    try:
                        await run_bridge(self.root, "evaluate", settings, connections, work,
                                         context.get("timeout_sec"))
                        grading = json.loads((work / "grading.json").read_text())
                        attempts.append({"attempt": attempt, "grading": grading,
                                         "model_calls": str(work / "model_calls.json")})
                        if grading["error"] is None and grading["metrics"] is not None:
                            break
                    except Exception as exc:
                        attempts.append({"attempt": attempt, "error_type": type(exc).__name__})
                        grading = None
            success = bool(grading and grading["error"] is None and grading["metrics"] is not None)
            raw = {"attempts": attempts, "native_result": grading["native_result"] if grading else None,
                   "native_status": session.get("status") if session else None,
                   "has_memo": bool(session and session.get("memo")),
                   "error": None if success else (grading or {}).get("error", "missing or mismatched native session")}
            if success:
                scored_session = dict(session, evaluation=grading["native_result"])
                evaluated_path = trial_dir / "session_evaluated.json"
                atomic_write_json(evaluated_path, scored_session)
                raw["summary_row"] = self.native_metrics.session_metrics(evaluated_path, scored_session, task.metadata["jurisdiction"])
            evaluated.trials[task.id] = VerifierResult(
                status="completed" if success else "failed", metrics=grading["metrics"] if success else {},
                raw_result=raw, artifacts={"session_evaluated": str(evaluated_path)} if success else {},
                provenance=VerifierProvenance(name=self.name, version="1.0", upstream_commit=UPSTREAM.commit,
                                              model=",".join(value.model for value in connections.values())),
            )
            logger.info("[evaluation] dlawbench %s/%s task=%s status=%s attempts=%s",
                        index + 1, len(tasks), task.id, evaluated.trials[task.id].status, len(attempts))
        evaluated.groups = self.build_groups(tasks, evaluated.trials)
        return evaluated

    def build_groups(self, tasks, results):
        rows = {key: result.raw_result["summary_row"] for key, result in results.items()
                if result.status == "completed"}
        groups = []
        for persona in dict.fromkeys(task.source.task_type for task in tasks):
            ids = [task.id for task in tasks if task.source.task_type == persona and task.id in rows]
            if ids:
                summary = self.native_metrics.build_summary_rows([rows[sid] for sid in ids], False)[0]
                groups.append({"task_type": persona, "task_ids": ids,
                               "metrics": {key: summary[key] for key in self.native_metrics.METRIC_KEYS},
                               "by_jurisdiction": self.native_metrics.build_summary_rows([rows[sid] for sid in ids], True)})
        return groups


class DLawBenchAggregator(BenchmarkAggregator):
    def aggregate(self, records, *, model_name):
        native = load_public_metrics(UPSTREAM.ensure(offline=True))
        tasks = mean_metrics(records)
        rows = []
        for record in records:
            evaluation = record.get("evaluation", {})
            if record.get("status") != "completed" or evaluation.get("status") != "completed":
                continue
            jurisdiction = record["source"]["provenance"]["jurisdiction"]
            rows.append({"model": model_name, "jurisdiction": jurisdiction,
                         "persona": record["task"], **evaluation["metrics"]})
        for persona, group in tasks.items():
            selected = [row for row in rows if row["persona"] == persona]
            if selected:
                summary = native.build_summary_rows(selected, False)[0]
                group["metrics"] = {key: summary[key] for key in native.METRIC_KEYS}
                group["by_jurisdiction"] = native.build_summary_rows(selected, True)
        return {"benchmark": "dlawbench", "model": model_name, "tasks": tasks,
                "overall": native.build_summary_rows(rows, False),
                "by_jurisdiction": native.build_summary_rows(rows, True)}

    def write_csv(self, summary, path, *, model_name):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["task", "model", "metrics", "score"])
            for persona, group in summary["tasks"].items():
                for metric, value in group["metrics"].items():
                    writer.writerow([persona, model_name, metric, value])
        return path
