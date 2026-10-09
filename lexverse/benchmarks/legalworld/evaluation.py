from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path

from lexverse.runtime.results import trial_relative_path
from lexverse.verifiers.base import TaskVerifier, BenchmarkAggregator, RunEvaluationResult, VerifierProvenance, VerifierResult, mean_metrics
from . import UPSTREAM
from .environment import connection_spec, run_native_process
from .tasks import ensure_dataset, stage_range

logger = logging.getLogger("lexverse.run")
STAGES = ["LC", "DRAFT", "CI", "SD", "APPEAL_DRAFT", "CIA"]


def has_scoring_error(value):
    if isinstance(value, dict):
        return bool(value.get("error") or value.get("status") in {"failed", "error"}
                    or (value.get("score") == 0 and str(value.get("reason", "")).startswith((
                        "Judge result missing", "Judge returned empty content",
                        "Judge returned non-dict payload:", "Judge payload missing expected metrics",
                        "Judge returned unrecognized payload:")))) or any(has_scoring_error(item) for item in value.values())
    if isinstance(value, list):
        return any(has_scoring_error(item) for item in value)
    return False


def scoring_coverage(pipeline, native):
    coverage = {}
    for stage in STAGES:
        if stage not in pipeline["stages_completed"]:
            coverage[stage] = "not_triggered"
            continue
        result = native.get("stage_eval_results", {}).get(stage, {})
        if stage == "SD" and result.get("status") == "not_implemented" and result.get("stage_score") is None:
            coverage[stage] = "not_implemented"
        elif has_scoring_error(result) or not isinstance(result.get("stage_score"), (int, float)) or not math.isfinite(result["stage_score"]):
            coverage[stage] = "scoring_failed"
        else:
            coverage[stage] = "scored"
    return coverage


class LegalWorldTaskVerifier(TaskVerifier):
    name = "legalworld"

    def __init__(self, *, offline=False):
        self.root = UPSTREAM.ensure_patched(Path(__file__).with_name("patches"), offline=offline)

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        context = context or {}
        connections = context["judge_connections"]
        evaluated = RunEvaluationResult(expected_task_ids=[task.id for task in tasks])
        for index, task in enumerate(tasks):
            record = records.get(task.id, {})
            if record.get("status") != "completed":
                continue
            trial = run_dir / "trials" / trial_relative_path(task.source.task_type, task.source.sample_id)
            path = Path(record.get("execution_artifacts", record.get("artifacts", {})).get("pipeline_result", trial / "pipeline_result.json"))
            if not path.is_absolute():
                path = run_dir / path
            native = None
            pipeline = None
            error = None
            coverage = {}
            work = trial / "evaluation"
            try:
                pipeline = json.loads(path.read_text())
                expected = stage_range(task.metadata["start_stage"], task.metadata["end_stage"])
                if (str(pipeline.get("case_id")) != task.source.sample_id
                        or pipeline.get("party_role") != task.metadata["party_role"]
                        or pipeline.get("stages_completed") != expected):
                    raise ValueError("pipeline identity or execution coverage mismatch")
                settings = {**task.metadata, "root": str(self.root.resolve()),
                            "dataset": str(ensure_dataset(offline=True)), "pipeline_result": str(path.resolve()),
                            "connections": {key: connection_spec(value) for key, value in connections.items()}}
                await run_native_process("evaluate", settings, connections, work, context.get("timeout_sec"))
                native = json.loads((work / "eval_result.json").read_text())
                coverage = scoring_coverage(pipeline, native)
            except Exception as exc:
                error = type(exc).__name__
                if (work / "eval_result.json").is_file():
                    native = json.loads((work / "eval_result.json").read_text())
                if pipeline:
                    coverage = scoring_coverage(pipeline, native or {})
            unsupported = bool(native and not error and coverage and all(
                value in {"not_implemented", "not_triggered"} for value in coverage.values()))
            success = bool(native and not error and not unsupported and all(value in {"scored", "not_implemented", "not_triggered"} for value in coverage.values()))
            metrics = {}
            if success:
                metrics = {key: native[key] for key in ("overall_score", "overall_possible_score", "overall_score_normalized")}
                metrics.update({f"{stage}.stage_score": result["stage_score"]
                                for stage, result in native["stage_eval_results"].items() if result.get("stage_score") is not None})
                success = all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
            raw = {"native_result": native, "coverage": coverage, "error_type": error,
                   "scoring_scope": "native implemented stages; SD excluded"}
            evaluated.trials[task.id] = VerifierResult(
                status="completed" if success else "not_applicable" if unsupported else "failed", metrics=metrics if success else {},
                raw_result=raw, artifacts={"native_eval": str(work / "eval_result.json")} if native else {},
                provenance=VerifierProvenance(name=self.name, version="1.0", upstream_commit=UPSTREAM.commit,
                                              model=connections["judge"].model),
            )
            logger.info("[evaluation] legalworld %s/%s task=%s status=%s coverage=%s",
                        index + 1, len(tasks), task.id, evaluated.trials[task.id].status, coverage)
        for kind in dict.fromkeys(task.source.task_type for task in tasks):
            ids = [task.id for task in tasks if task.source.task_type == kind and task.id in evaluated.trials
                   and evaluated.trials[task.id].status == "completed"]
            if ids:
                common = set.intersection(*(set(evaluated.trials[sid].metrics) for sid in ids))
                evaluated.groups.append({"task_type": kind, "task_ids": ids,
                                         "metrics": {key: sum(evaluated.trials[sid].metrics[key] for sid in ids) / len(ids) for key in common}})
        return evaluated


class LegalWorldAggregator(BenchmarkAggregator):
    def aggregate(self, records, *, model_name):
        return {"benchmark": "legalworld", "model": model_name, "tasks": mean_metrics(records),
                "scoring_scope": "native implemented stages; SD not_implemented"}

    def write_csv(self, summary, path, *, model_name):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["task", "model", "metrics", "score"])
            for kind, group in summary["tasks"].items():
                for key, value in group["metrics"].items():
                    writer.writerow([kind, model_name, key, value])
        return path
