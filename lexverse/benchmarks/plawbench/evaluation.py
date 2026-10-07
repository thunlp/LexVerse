"""Inject a run-scoped judge into the native evaluator; retain native metrics."""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import threading
import math
from pathlib import Path

from lexverse.verifiers.base import (
    TaskVerifier, BenchmarkAggregator, RunEvaluationResult, VerifierProvenance, VerifierResult, mean_metrics,
)
from . import UPSTREAM


def load_evaluator(path):
    spec = importlib.util.spec_from_file_location("lexverse_plawbench_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_grading(task_type, parsed):
    """Reject missing/non-numeric scores before upstream's permissive defaults."""
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("judge output contains no grading JSON object")
    if task_type == "case_analysis_merge":
        details = parsed.get("score_details", {})
        pairs = [details.get(key, {}) for key in ("法条依据", "案件事实", "推理过程", "结论")]
    else:
        pairs = [parsed]
    for pair in pairs:
        for key in ("total_points", "max_points"):
            value = pair.get(key)
            if isinstance(value, bool) or value is None:
                raise ValueError(f"judge grading is missing numeric {key}")
            try:
                valid = math.isfinite(float(value))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"judge grading has invalid {key}")

class PLawBenchTaskVerifier(TaskVerifier):
    name = "plawbench"

    def __init__(self, *, offline=False, upstream=UPSTREAM):
        self.root = upstream.ensure_patched(Path(__file__).parent / "patches", offline=offline)
        self.version = upstream.commit
        self.native = load_evaluator(self.root / "prompt_zh.py")

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        task = tasks[0]
        result = await self.evaluate_task(task, records[task.id], run_dir=run_dir,
                                          model_name=model_name, context=context)
        return RunEvaluationResult(trials={task.id: result},
                                   groups=self.build_groups(tasks, {task.id: result}),
                                   expected_task_ids=[task.id])

    async def evaluate_task(self, task, record, *, run_dir, model_name, context):
        judge = context["judge_provider"]
        attempts = context.get("max_attempts", 3)
        native = self.native
        loop = asyncio.get_running_loop()
        history = []
        result = None
        item = dict(task.evaluation.config["item"], response=record.get("response", ""))
        scorer = task.evaluation.config["task_type"]
        for attempt in range(1, attempts + 1):
            evidence = {"attempt": attempt, "raw_judge_output": None}
            pending = []
            cancelled = threading.Event()

            def call_judge(messages):
                if cancelled.is_set():
                    raise RuntimeError("scoring cancelled before judge request")
                future = asyncio.run_coroutine_threadsafe(judge.generate(messages), loop)
                pending.append(future)
                if cancelled.is_set():
                    future.cancel()
                response = future.result()
                evidence["raw_judge_output"] = response.content
                parsed = native.parse_json_to_dict(response.content)
                validate_grading(scorer, parsed)
                return response.content

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(native.LegalEvaluator(call_judge).evaluate, item, scorer),
                    timeout=context.get("timeout_sec"),
                )
                if result["metrics"].get("failed") or not all(math.isfinite(v) for v in result["metrics"].values()):
                    raise ValueError("native evaluator returned invalid metrics")
                evidence["parsed_judge_output"] = result["raw_judge_output"]
                history.append(evidence)
                break
            except asyncio.CancelledError:
                cancelled.set()
                for future in pending:
                    future.cancel()
                raise
            except Exception as exc:
                # Keep actionable failure details without persisting connection credentials.
                message = str(exc)
                for secret in (getattr(judge, "api_key", None),):
                    if secret:
                        message = message.replace(secret, "[redacted]")
                evidence.update(error_type=type(exc).__name__, error=message)
                history.append(evidence)
                result = None
            finally:
                cancelled.set()
                for future in pending:
                    future.cancel()
        success = result is not None
        raw = {"attempts": history, "native_result": result, "source_score": task.raw_payload.get("score"),
               "judge_score_input": item.get("score"), "score_source": task.metadata["score_source"]}
        verified = VerifierResult(
            status="completed" if success else "failed",
            metrics=result["metrics"] if success else {"failed": 1}, raw_result=raw,
            provenance=VerifierProvenance(name=self.name, version="1.0", upstream_commit=self.version,
                                          model=context.get("judge_model")),
        )
        return verified

    def build_groups(self, tasks, results):
        groups = []
        for kind in dict.fromkeys(task.source.task_type for task in tasks):
            ids = [task.id for task in tasks if task.source.task_type == kind
                   and task.id in results and results[task.id].status == "completed"]
            if ids:
                metrics = results[ids[0]].metrics
                groups.append({"task_type": kind, "task_ids": ids,
                                         "metrics": {key: sum(results[sid].metrics[key] for sid in ids) / len(ids)
                                                     for key in metrics}})
        return groups


class PLawBenchAggregator(BenchmarkAggregator):
    def aggregate(self, records, *, model_name):
        # No executable upstream aggregate formula is supplied. Report per-task means.
        tasks = mean_metrics(records)
        documents = []
        for record in records:
            if record.get("task") != "document_generation":
                continue
            role = (record.get("source", {}).get("provenance", {}).get("document_role")
                    or str(record.get("sample_id", "")).partition(":")[0])
            if role in {"plaintiff", "defendant"}:
                documents.append(dict(record, task=role))
        if "document_generation" in tasks:
            tasks["document_generation"]["subtypes"] = mean_metrics(documents)
        return {"benchmark": "plawbench", "model": model_name, "tasks": tasks}

    def write_csv(self, summary, path, *, model_name):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["task", "model", "metrics", "score"])
            for kind, group in summary["tasks"].items():
                for metric, value in group["metrics"].items():
                    writer.writerow([kind, model_name, metric, value])
        return path
