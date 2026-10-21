"""Invoke public formatter/scorers and aggregate native counts, not per-case F1."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from lexverse.runtime.results import trial_relative_path
from lexverse.verifiers.base import (
    BenchmarkAggregator,
    RunEvaluationResult,
    TaskVerifier,
    VerifierProvenance,
    VerifierResult,
    mean_metrics,
)
import threading
from . import UPSTREAM
from .tasks import KINDS, PATCHES


def load_public_metrics(root, name):
    path = root / "AgentsCourt" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lexverse_simucourt_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gold_row(task):
    reference = task.evaluation.reference
    case_types = {value: f"{key}案件" for key, value in KINDS.items()}
    return {
        "案件类型": case_types[task.metadata["category"]],
        "审判依据": reference["laws"],
        **reference["judgment"],
    }


def group_metrics(rows, expected=None):
    valid = [row for row in rows if row.get("deterministic_scored")]
    if expected is not None and len(valid) != expected:
        return {}
    judged = [row for row in valid if row.get("judge_status") == "scored"]
    if expected is not None and any(
        row.get("judge_status") not in {"scored", "not_applicable"} for row in valid
    ):
        return {}
    metrics = {}
    if valid:
        root = UPSTREAM.ensure_patched(PATCHES, offline=True)
        gold = {str(i): row["gold"] for i, row in enumerate(valid)}
        predicted = {str(i): row["prediction"] for i, row in enumerate(valid)}
        law_scores = load_public_metrics(root, "evaluation_laws").score(gold, predicted)
        metrics.update({f"law_micro_{key}": value for key, value in law_scores.items()})
        criminal = {
            key: value for key, value in gold.items()
            if value["案件类型"] == "刑事案件"
        }
        if criminal:
            metrics.update(load_public_metrics(root, "evaluation_crimial").score(criminal, predicted))
        if judged:
            counts = [row["judge_counts"] for row in judged]
            other_scores = load_public_metrics(root, "evaluation_other").aggregate_counts(counts)
            metrics.update({f"judgment_micro_{key}": value for key, value in other_scores.items()})
    return metrics


def evaluation_paths(task, run_dir):
    rel_path = trial_relative_path(task.source.task_type, task.source.sample_id)
    directory = Path(run_dir) / "trials" / rel_path / "evaluation"
    return {name: directory / f"{name}.json" for name in ("formatting", "grading", "model_calls")}


class SimuCourtTaskVerifier(TaskVerifier):
    name = "simucourt"

    def __init__(self, *, offline=False):
        self.root = UPSTREAM.ensure_patched(PATCHES, offline=offline)

    def prepare_unit(self, tasks, records, *, run_dir, model_name, context):
        paths = evaluation_paths(tasks[0], run_dir)
        hashes = context.get("resume_artifact_hashes", {})
        if not context.get("resume_native") or not all(
            path.is_file() and hashes.get(str(path.resolve())) == hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items() if name in {"formatting", "model_calls"}
        ):
            for path in paths.values():
                path.unlink(missing_ok=True)

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        task = tasks[0]
        result = await self.evaluate_task(task, records[task.id], run_dir=run_dir, context=context)
        return RunEvaluationResult(
            trials={task.id: result},
            groups=self.build_groups(tasks, {task.id: result}),
            expected_task_ids=[task.id],
        )

    async def evaluate_task(self, task, record, *, run_dir, context):
        other = task.metadata["category"] != "criminal"
        raw = {
            "deterministic_scored": False,
            "judge_status": "missing" if other else "not_applicable",
            "format_status": "missing",
        }
        metrics, status = {}, "failed"
        paths = evaluation_paths(task, run_dir)
        judge = context["judge_provider"]
        other_judge = context.get("other_judge_provider", judge)
        loop = asyncio.get_running_loop()
        calls = []
        prediction = None
        if context.get("resume_native") and paths["formatting"].is_file():
            saved = json.loads(paths["formatting"].read_text(encoding="utf-8"))
            if saved.get("status") == "completed":
                prediction = saved["prediction"]
                calls = json.loads(paths["model_calls"].read_text(encoding="utf-8"))
        formatter = load_public_metrics(self.root, "judge_result_process")
        native_other = load_public_metrics(self.root, "evaluation_other")
        category_cn = next(key for key, value in KINDS.items() if value == task.metadata["category"])
        case = {
            "案号": task.source.sample_id,
            "类别": category_cn,
            "一审法院审判结果": task.evaluation.reference.get("first_judgment", ""),
        }
        gold = gold_row(task)
        if record.get("status") == "completed":
            for attempt in range(1, context.get("max_attempts", 3) + 1):
                pending = []
                cancelled = threading.Event()
                phase = "formatting" if prediction is None else "judgment"

                def call_judge(*args):
                    if cancelled.is_set():
                        raise RuntimeError("scoring cancelled before request")
                    messages = [
                        {"role": "system", "content": "你是一个专业的法律助手"},
                        {"role": "user", "content": args[-1]},
                    ]
                    active_judge = other_judge if phase == "judgment" else judge
                    evidence = {"attempt": attempt, "phase": phase, "messages": messages}
                    calls.append(evidence)
                    future = asyncio.run_coroutine_threadsafe(active_judge.generate(messages), loop)
                    pending.append(future)
                    if cancelled.is_set():
                        future.cancel()
                    try:
                        response = future.result()
                        evidence["output"] = response.content
                        return response.content
                    except Exception as exc:
                        evidence["error_type"] = type(exc).__name__
                        raise

                try:
                    if prediction is None:
                        prediction = await asyncio.wait_for(
                            asyncio.to_thread(
                                formatter.format_prediction,
                                case, record.get("response", ""),
                                task.metadata["stage"], call_judge,
                            ),
                            timeout=context.get("timeout_sec"),
                        )
                        paths["formatting"].parent.mkdir(parents=True, exist_ok=True)
                        paths["formatting"].write_text(
                            json.dumps({"status": "completed", "prediction": prediction}, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    raw.update(
                        format_status="completed",
                        deterministic_scored=True,
                        gold=gold,
                        prediction=prediction,
                    )
                    if other:
                        phase = "judgment"
                        native_other.GPT_4 = call_judge
                        sid = task.source.sample_id
                        _, counts = await asyncio.wait_for(
                            asyncio.to_thread(
                                native_other.score,
                                [case], {sid: gold}, {sid: prediction}, task.metadata["stage"],
                            ),
                            timeout=context.get("timeout_sec"),
                        )
                        if sid not in counts:
                            raw["judge_status"] = "invalid" if calls and "output" in calls[-1] else "failed"
                            raise ValueError("native judgment scoring failed")
                        raw.update(judge_status="scored", judge_counts=counts[sid])
                    status = "completed"
                    break
                except asyncio.CancelledError:
                    cancelled.set()
                    for future in pending:
                        future.cancel()
                    raise
                except Exception as exc:
                    raw["error_type"] = type(exc).__name__
                    if prediction is None:
                        is_invalid = isinstance(exc, (ValueError, SyntaxError, TypeError))
                        raw["format_status"] = "invalid" if is_invalid else "failed"
                        status = "invalid" if raw["format_status"] == "invalid" else "failed"
                    elif other and raw["judge_status"] != "invalid":
                        raw["judge_status"] = "failed"
                finally:
                    cancelled.set()
                    for future in pending:
                        future.cancel()
        if raw["deterministic_scored"]:
            metrics = group_metrics([raw])
        paths["grading"].parent.mkdir(parents=True, exist_ok=True)
        if not paths["formatting"].exists():
            paths["formatting"].write_text(json.dumps({"status": raw["format_status"]}), encoding="utf-8")
        paths["grading"].write_text(
            json.dumps({"status": status, "raw_result": raw, "metrics": metrics}, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["model_calls"].write_text(json.dumps(calls, ensure_ascii=False), encoding="utf-8")
        return VerifierResult(
            status=status,
            metrics=metrics,
            raw_result=raw,
            artifacts={name: str(path) for name, path in paths.items()},
            provenance=VerifierProvenance(
                name=self.name, version="3.0",
                upstream_commit=UPSTREAM.commit, model=context.get("judge_model"),
            ),
        )

    def build_groups(self, tasks, results):
        groups = []
        for kind in dict.fromkeys(task.source.task_type for task in tasks):
            selected = [task for task in tasks if task.source.task_type == kind]
            completed_ids = [
                task.id for task in selected
                if task.id in results and results[task.id].status == "completed"
            ]
            rows = [results[task_id].raw_result for task_id in completed_ids]
            groups.append({
                "task_type": kind,
                "task_ids": completed_ids,
                "metrics": group_metrics(rows, len(selected)) if len(completed_ids) == len(selected) else {},
            })
        return groups

    def write_results(self, result, tasks, records, *, run_dir, model_name):
        scored = [
            dict(
                records.get(task.id) or {"id": task.id, "status": "missing"},
                task=task.source.task_type,
                **({"evaluation": result.trials[task.id].model_dump(mode="json")}
                   if task.id in result.trials else {}),
            )
            for task in tasks
        ]
        aggregator = SimuCourtAggregator()
        summary = aggregator.aggregate(scored, model_name=model_name)
        aggregator.write_csv(summary, Path(run_dir) / "evaluation_result.csv", model_name=model_name)


class SimuCourtAggregator(BenchmarkAggregator):
    def aggregate(self, records, *, model_name):
        scored = [
            {
                **record,
                "evaluation": record.get("evaluation") or {"status": "pending", "metrics": {}},
            }
            for record in records
        ]
        tasks = mean_metrics(scored)
        rows_by_kind = {}
        for kind, group in tasks.items():
            selected = [record for record in scored if record["task"] == kind]
            rows = [(record.get("evaluation") or {}).get("raw_result", {}) for record in selected]
            rows_by_kind[kind] = rows
            group["metrics"] = (
                group_metrics(rows, len(selected))
                if group["sample_count"] == group["expected_samples"] else {}
            )
        stages = {}
        for stage in ("first", "second"):
            stage_records = [
                {**record, "task": stage}
                for record in scored
                if str(record.get("task", "")).startswith(f"{stage}_")
            ]
            if stage_records:
                stage_summary = mean_metrics(stage_records)[stage]
                rows = [
                    row for kind, items in rows_by_kind.items()
                    if kind.startswith(f"{stage}_") for row in items
                ]
                stage_summary["metrics"] = (
                    group_metrics(rows, len(rows))
                    if stage_summary["sample_count"] == stage_summary["expected_samples"] else {}
                )
                stages[stage] = stage_summary
        return {
            "benchmark": "simucourt",
            "model": model_name,
            "tasks": tasks,
            "stages": stages,
        }

    def write_csv(self, summary, path, *, model_name):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["task", "model", "metrics", "score"])
            for kind, group in summary["tasks"].items():
                for metric, value in group["metrics"].items():
                    writer.writerow([kind, model_name, metric, value])
        return path
