"""Invoke patched MILE Goal and Interaction evaluators and aggregate results."""
from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from lexverse.verifiers.base import (
    BenchmarkAggregator,
    RunEvaluationResult,
    TaskVerifier,
    VerifierProvenance,
    VerifierResult,
    mean_metrics,
)
from lexverse.runtime.errors import EvaluationError, TrialProcessError, TrialTimeoutError
from lexverse.runtime.executor import run_subprocess
from lexverse.runtime.results import artifact_model_name, atomic_write_json, execution_artifacts_valid
from lexverse.runtime.sources import UpstreamSource
from lexverse.tasks.schema import LexVerseTask

from . import UPSTREAM
from .tasks import TASK_TYPE, validate_history


@dataclass
class MILEEvalConfig:
    scoring_api_key: str
    scoring_api_base: str | None = None
    scoring_model: str = "gpt-4o-mini"
    scoring_parameters: dict[str, Any] = field(default_factory=dict)
    request_timeout_sec: float | None = None
    max_attempts: int = 3
    concurrency: int = 4


@dataclass
class MILEEvalResult:
    task_type: str
    model_name: str
    intermediate_dir: Path
    final_json: Path
    per_case: list[dict]
    aggregate: dict
    process_error: dict | None = None


GOAL_KEYS = (
    "Fact_and_Reason", "Claim", "Evidence", "Format", "Professionality",
    "ave_format_prof", "plaintiff_accuracy", "defendant_accuracy",
)
INTERACTION_KEYS = ("互动性", "专业性", "逻辑性")


def _case_id(value, path):
    return str(value.get("case_id", path.stem))


def _metrics(value: dict, expected) -> dict[str, float]:
    goal = value.get("official_score", {}).get("Goal", {})
    interaction = value.get("official_raw", {}).get("Interaction", {})
    if not goal or not interaction:
        return {}
    axes = {
        "Fact_and_Reason": "F_R",
        "Claim": "CLA",
        "Evidence": "EVID",
        "Format": "STA",
        "Professionality": "G_PROF",
    }
    interaction_axes = {"互动性": "INT", "专业性": "I_PROF", "逻辑性": "LOGI"}
    metrics = {name: goal[key] * 10 for key, name in axes.items()}
    metrics.update(
        CLI=goal["plaintiff_accuracy"] * 100,
        DEF=goal["defendant_accuracy"] * 100,
    )
    windows = [window[index] for index, window in interaction.items()]
    metrics.update({
        name: sum(window[key] for window in windows) / len(windows) * 10
        for key, name in interaction_axes.items()
    })
    metrics["G_AVE"] = sum(metrics[key] for key in ("CLI", "DEF", *axes.values())) / 7
    metrics["I_AVE"] = sum(metrics[key] for key in interaction_axes.values()) / 3
    metrics["Total"] = (metrics["G_AVE"] + metrics["I_AVE"]) / 2
    return {name: metrics[name] for name in expected if name in metrics}


def _finite_score(value, low=0, high=10):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _parsed_score(response, key):
    try:
        parsed = json.loads(response)
    except (ValueError, TypeError):
        raise ValueError("invalid_json") from None
    if not isinstance(parsed, dict) or key not in parsed:
        raise ValueError("missing_key")
    if type(parsed[key]) is not int or not 0 <= parsed[key] <= 10:
        raise ValueError("invalid_score")
    if not isinstance(parsed.get("解释"), str) or not parsed["解释"].strip():
        raise ValueError("empty_explanation")
    return parsed[key]


def _complete_case(value):
    try:
        if (not isinstance(value, dict)
                or value.get("status") != "completed"
                or value.get("process_errors")):
            return False
        goal = value["official_score"]["Goal"]
        raw_goal = value["official_raw"]["Goal"]
        interaction = value["official_raw"]["Interaction"]
        if not isinstance(goal, dict) or set(goal) != set(GOAL_KEYS):
            return False
        if not all(
            _finite_score(goal.get(key), high=1 if key.endswith("accuracy") else 10)
            for key in GOAL_KEYS
        ):
            return False
        for expr, key in (
            (raw_goal["score_claim"], "分数"),
            (raw_goal["score_r_f"], "分数"),
            (raw_goal["score_evidence"], "分数"),
            (raw_goal["score_globe"]["模板"], "规范性"),
            (raw_goal["score_globe"]["专业"], "专业性"),
        ):
            _parsed_score(expr, key)
        expected = value["expected_window_ids"]
        if not expected or not isinstance(interaction, dict) or set(interaction) != set(expected):
            return False
        for index in expected:
            window = interaction[index][index]
            if not all(type(window.get(key)) is int and _finite_score(window[key]) for key in INTERACTION_KEYS):
                return False
        metadata = value.get("request_metadata", {})
        for mode, count in (("Goal", 5), ("Interaction", len(expected))):
            meta = metadata.get(mode)
            if not isinstance(meta, dict) or len(meta.get("responses", ())) != count:
                return False
            for resp in meta["responses"]:
                if resp.get("finish_reason") != "stop" or resp.get("explanation_chars", 0) <= 0:
                    return False
        return True
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def _prepare_intermediate(
    directory: Path, tasks, *, resume,
    resume_task_ids=None, artifact_hashes=None,
):
    """Keep readable native cases with matching inputs and saved artifact hashes."""
    expected = {
        task.source.sample_id for task in tasks
        if resume_task_ids is None or task.id in resume_task_ids
    }
    for path in directory.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            saved_hash = (artifact_hashes or {}).get(str(path.resolve()))
            valid = (
                (resume or resume_task_ids is not None)
                and _complete_case(value)
                and _case_id(value, path) in expected
                and saved_hash is not None
                and hashlib.sha256(path.read_bytes()).hexdigest() == saved_hash
            )
            for filename in value.get("native_artifacts", {}).values():
                artifact = Path(filename)
                digest = (artifact_hashes or {}).get(str(artifact.resolve()))
                valid = (
                    valid and digest is not None and artifact.is_file()
                    and hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
                )
        except (OSError, ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            path.unlink()


def _read_per_case(int_dir: Path) -> list[dict]:
    if not int_dir.exists():
        return []
    out = []
    for path in sorted(int_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and _case_id(value, path) == path.stem:
                out.append(value)
        except (OSError, ValueError, TypeError):
            continue
    return out


def load_generation_artifacts(task, record, run_dir):
    declared = record.get("execution_artifacts", record.get("artifacts", {}))
    if not execution_artifacts_valid({**record, "execution_artifacts": declared}, Path(run_dir)):
        raise ValueError("MILE generation artifact missing or hash mismatch")
    sid = int(task.source.sample_id)
    session = json.loads((Path(run_dir) / declared["session"]).read_text(encoding="utf-8"))
    if session.get("status") != "completed" or session.get("case_id") != sid:
        raise ValueError("MILE session/task mismatch or incomplete generation")
    history = validate_history(session.get("dialog_history"))
    for name in ("dialog_history", "native_all_info"):
        lines = (Path(run_dir) / declared[name]).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        if len(rows) != 1 or rows[0].get("case_id") != sid or rows[0].get("dialog_history") != history:
            raise ValueError("MILE native JSONL/session mismatch")
    if record.get("response") != history[-1]["content"]:
        raise ValueError("MILE response/native final complaint mismatch")
    return session, history


class MILEEvaluator:
    def __init__(
        self, task_type: str = TASK_TYPE, upstream: UpstreamSource = UPSTREAM,
        offline: bool = False, python_executable: str = sys.executable,
        case_database: Path | None = None,
    ):
        self.task_type = task_type
        self._python = python_executable
        self._cache = upstream.ensure_patched(Path(__file__).parent / "patches", offline=offline)
        self._eval_py = self._cache / "src" / "evaluation" / "src_eval" / "main.py"
        if not self._eval_py.is_file():
            raise EvaluationError(f"upstream missing MILE evaluator: {self._eval_py}")
        if case_database is None or not Path(case_database).is_file():
            raise EvaluationError("MILE verification requires an explicit gold case database")
        self._case_database = Path(case_database).resolve()

    async def evaluate(
        self, model_name: str, dialog_history_path: Path,
        output_dir: Path, config: MILEEvalConfig, *,
        timeout_sec: float | None = None,
    ) -> MILEEvalResult:
        model_tag = artifact_model_name(model_name)
        output_dir = output_dir.resolve()
        intermediate = output_dir / "intermediate" / model_tag / self.task_type
        intermediate.mkdir(parents=True, exist_ok=True)
        final_json = output_dir / "final" / f"{model_tag}_final.json"
        lines = dialog_history_path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        gold = json.loads(self._case_database.read_text(encoding="utf-8"))
        if not rows or len({row["case_id"] for row in rows}) != len(rows):
            raise ValueError("MILE evaluation requires unique case IDs")
        if len({item["id"] for item in gold}) != len(gold) or {item["id"] for item in gold} != {row["case_id"] for row in rows}:
            raise ValueError("MILE gold/task ID mismatch")

        env = dict(os.environ)
        env.pop("OPENAI_API_BASE", None)
        if config.scoring_api_base:
            env["OPENAI_API_BASE"] = config.scoring_api_base
        request_timeout = config.request_timeout_sec if config.request_timeout_sec is not None else 60
        env.update(
            OPENAI_API_KEY=config.scoring_api_key,
            MASER_SCORING_MODEL=config.scoring_model,
            MASER_SCORING_PARAMETERS=json.dumps(config.scoring_parameters),
            MASER_REQUEST_TIMEOUT=str(request_timeout),
            MASER_MAX_ATTEMPTS=str(config.max_attempts),
            PYTHONUNBUFFERED="1",
        )
        semaphore = asyncio.Semaphore(config.concurrency)

        async def run_mode(row, mode):
            case_id = row["case_id"]
            work = output_dir / "cases" / str(case_id) / mode
            if work.exists():
                shutil.rmtree(work)
            (work / "data").mkdir(parents=True)
            (work / "data" / "case.jsonl").write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            reference = [item for item in gold if item["id"] == case_id]
            atomic_write_json(work / "gold.json", reference)
            native = work / "output" / mode / "case" / "case.json"
            error = None
            async with semaphore:
                try:
                    await run_subprocess(
                        command=[
                            self._python, str(self._eval_py),
                            "--tasks", mode,
                            "--input_dir", "data",
                            "--output_dir", "output",
                            "--gold", "gold.json",
                        ],
                        work_dir=work, timeout_sec=timeout_sec, env=env,
                    )
                except (TrialProcessError, TrialTimeoutError) as exc:
                    error = {
                        "error_type": type(exc).__name__,
                        "returncode": getattr(exc, "returncode", None),
                    }
                    atomic_write_json(work / "error.json", error)
            artifacts = {
                str(path.relative_to(work)): str(path)
                for path in work.rglob("*") if path.is_file()
            }

            def read_json(path):
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    return None

            raw_json = Path(str(native) + ".raw.json")
            return (
                mode, read_json(native), read_json(raw_json),
                read_json(work / "request_metadata.json"), error, artifacts,
            )

        async def run_case(row):
            case_id = row["case_id"]
            case_path = intermediate / f"{case_id}.json"
            if case_path.is_file():
                try:
                    saved = json.loads(case_path.read_text(encoding="utf-8"))
                    if (
                        _complete_case(saved)
                        and saved.get("native_artifacts")
                        and all(Path(path).is_file() for path in saved["native_artifacts"].values())
                    ):
                        return saved
                except (OSError, ValueError, TypeError):
                    pass
            history = validate_history(row["dialog_history"])
            turns = history[-1]["turn"]
            expected = [str(index) for index in range(2, turns, 2)]
            if turns >= 2 and turns % 2 == 0:
                expected.append(str(turns))
            outputs = await asyncio.gather(*(run_mode(row, mode) for mode in ("Goal", "Interaction")))
            value = {
                "case_id": case_id,
                "status": "failed",
                "turns": turns,
                "expected_window_ids": expected,
                "official_raw": {},
                "official_score": {},
                "request_metadata": {},
                "process_errors": {},
                "native_artifacts": {},
            }
            for mode, native, raw, metadata, error, artifacts in outputs:
                value["request_metadata"][mode] = metadata
                value["native_artifacts"].update({f"{mode}/{name}": path for name, path in artifacts.items()})
                if error:
                    value["process_errors"][mode] = error
                if mode == "Goal":
                    value["official_score"][mode] = native
                    value["official_raw"][mode] = raw[0] if isinstance(raw, list) and len(raw) == 1 else None
                else:
                    windows = native.get(str(case_id)) if isinstance(native, dict) else None
                    value["official_score"][mode] = windows
                    value["official_raw"][mode] = windows
            windows = value["official_raw"].get("Interaction")
            scored = len(set(windows if isinstance(windows, dict) else {}) & set(expected))
            value["window_coverage"] = {
                "expected": len(expected),
                "scored": scored,
                "failed": len(expected) - scored,
                "missing": 0,
            }
            value["status"] = "completed" if _complete_case({**value, "status": "completed"}) else "failed"
            atomic_write_json(case_path, value)
            return value

        per_case = await asyncio.gather(*(run_case(row) for row in rows))
        complete = [value for value in per_case if _complete_case(value)]
        success = len(complete) == len(rows)
        coverage = {
            mode: {
                "expected": len(rows),
                "scored": len(complete),
                "failed": len(rows) - len(complete),
                "missing": 0,
            }
            for mode in ("Goal", "Interaction")
        }
        goal_aggregate = {
            key: sum(value["official_score"]["Goal"][key] for value in complete) / len(complete)
            for key in GOAL_KEYS
        } if success else None
        interaction_aggregate = {
            str(value["case_id"]): value["official_raw"]["Interaction"]
            for value in complete
        }
        aggregate = {
            "status": "completed" if success else "failed",
            "coverage": coverage,
            "official_raw": {"Goal": goal_aggregate, "Interaction": interaction_aggregate},
        }
        atomic_write_json(final_json, aggregate)
        return MILEEvalResult(self.task_type, model_name, intermediate, final_json, per_case, aggregate)


class MILETaskVerifier(TaskVerifier):
    name = "maser"
    grouped = True

    def prepare_unit(self, tasks, records, *, run_dir, model_name, context):
        task_type = tasks[0].source.task_type
        model_tag = artifact_model_name(model_name)
        directory = Path(run_dir) / "trials" / task_type / "verifier" / "intermediate" / model_tag / task_type
        eligible = [task for task in tasks if records.get(task.id, {}).get("status") == "completed"]
        _prepare_intermediate(
            directory, eligible, resume=context.get("resume_native", False),
            resume_task_ids=context.get("resume_task_ids"),
            artifact_hashes=context.get("resume_artifact_hashes"),
        )

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        from collections import defaultdict

        context = dict(context or {})
        grouped = defaultdict(list)
        for task in tasks:
            grouped[task.source.task_type].append(task)
        result = RunEvaluationResult(expected_task_ids=[task.id for task in tasks])
        for task_type, scenario_tasks in grouped.items():
            completed_tasks, dialog_paths = [], []
            for task in scenario_tasks:
                record = records.get(task.id, {})
                if record.get("status") != "completed":
                    continue
                try:
                    load_generation_artifacts(task, record, run_dir)
                    completed_tasks.append(task)
                    declared = record.get("execution_artifacts", record.get("artifacts", {}))
                    dialog_paths.append(Path(run_dir) / declared["dialog_history"])
                except (ValueError, KeyError, TypeError, OSError) as error:
                    result.trials[task.id] = VerifierResult(
                        status="invalid",
                        raw_result={"error_type": type(error).__name__},
                        provenance=VerifierProvenance(
                            name=self.name, version="5.0", upstream_commit=task.source.version,
                        ),
                    )
            if not completed_tasks:
                continue
            verified, aggregate = await self.verify_scenario(
                completed_tasks, dialog_paths,
                output_dir=Path(run_dir) / "trials" / task_type / "verifier",
                context={"model_name": model_name, **context},
            )
            for task in completed_tasks:
                result.trials[task.id] = verified[task.source.sample_id]
            groups = self.build_groups(scenario_tasks, result.trials)
            first_verified = verified[completed_tasks[0].source.sample_id]
            for group in groups:
                group.update(
                    granularity="scenario",
                    expected_task_ids=[task.id for task in scenario_tasks],
                    eligible_task_ids=[task.id for task in completed_tasks],
                    raw_result=aggregate,
                    artifacts=first_verified.artifacts,
                    provenance=first_verified.provenance.model_dump(mode="json"),
                )
                if aggregate.get("status") != "completed":
                    group["metrics"] = {}
            result.groups.extend(groups)
        return result

    def collect_partial_unit(self, tasks, records, *, run_dir, model_name, context):
        result = RunEvaluationResult(expected_task_ids=[task.id for task in tasks])
        model_tag = artifact_model_name(model_name)
        for task_type, scenario_tasks in self.evaluation_units(tasks).items():
            output_dir = Path(run_dir) / "trials" / task_type / "verifier"
            intermediate = output_dir / "intermediate" / model_tag / task_type
            eligible = [
                task for task in scenario_tasks
                if records.get(task.id, {}).get("status") == "completed"
            ]
            per_case = [
                value for value in _read_per_case(intermediate)
                if _complete_case(value) and value.get("native_artifacts")
                and all(Path(path).is_file() for path in value["native_artifacts"].values())
            ]
            aggregate = {"status": "incomplete"}
            final_json = output_dir / "final" / f"{model_tag}_final.json"
            evaluated = MILEEvalResult(
                task_type, model_name, intermediate, final_json, per_case, aggregate,
            )
            verified = self._case_results(eligible, evaluated, context, include_final=False)
            result.trials.update({
                task.id: verified[task.source.sample_id]
                for task in eligible
                if verified[task.source.sample_id].status == "completed"
            })
            groups = self.build_groups(scenario_tasks, result.trials)
            for group in groups:
                group.update(
                    granularity="scenario",
                    eligible_task_ids=[task.id for task in eligible],
                    raw_result=aggregate,
                    metrics={},
                )
            result.groups.extend(groups)
        return result

    def write_results(self, result, tasks, records, *, run_dir, model_name):
        scored_records = []
        for task in tasks:
            record = dict(records.get(task.id) or {"id": task.id, "status": "missing"})
            record.setdefault("task", task.source.task_type)
            if task.id in result.trials:
                verified = result.trials[task.id]
                record["evaluation"] = verified.model_dump(mode="json")
                if verified.status != "completed":
                    record["status"] = "failed"
            scored_records.append(record)
        aggregator = MILEAggregator()
        summary = aggregator.aggregate(scored_records, model_name=model_name)
        for group in result.groups:
            summary["tasks"][group["task_type"]]["metrics"] = group["metrics"]
        aggregator.write_csv(summary, Path(run_dir) / "evaluation_result.csv", model_name=model_name)

    async def verify_scenario(
        self, tasks: list[LexVerseTask], dialog_paths: list[Path], *,
        output_dir: Path, context: dict | None = None,
    ) -> tuple[dict[str, VerifierResult], dict]:
        context = context or {}
        task_type = tasks[0].source.task_type
        api_key = context.get("scoring_api_key")
        if not api_key:
            raise ValueError("MILE verification requires context.scoring_api_key")
        output_dir.mkdir(parents=True, exist_ok=True)
        case_database = output_dir / "gold.json"
        atomic_write_json(case_database, [task.evaluation.reference for task in tasks])
        evaluator = MILEEvaluator(
            task_type,
            offline=context.get("offline", False),
            python_executable=context.get("python_executable", sys.executable),
            case_database=case_database,
        )
        model_tag = artifact_model_name(context.get("model_name", "LexVerse"))
        intermediate = output_dir / "intermediate" / model_tag / task_type
        _prepare_intermediate(
            intermediate, tasks, resume=context.get("resume_native", False),
            resume_task_ids=context.get("resume_task_ids"),
            artifact_hashes=context.get("resume_artifact_hashes"),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False, dir=output_dir,
        ) as combined:
            combined_path = Path(combined.name)
            for path in dialog_paths:
                text = path.read_text(encoding="utf-8")
                combined.write(text)
                if text and not text.endswith("\n"):
                    combined.write("\n")
        try:
            eval_config = MILEEvalConfig(
                scoring_api_key=api_key,
                scoring_api_base=context.get("scoring_api_base"),
                scoring_model=context.get("scoring_model", "gpt-4o-mini"),
                scoring_parameters=context.get("scoring_parameters", {}),
                request_timeout_sec=context.get("request_timeout_sec"),
                max_attempts=context.get("max_attempts", 3),
                concurrency=context.get("concurrency", 4),
            )
            evaluated = await evaluator.evaluate(
                model_name=context.get("model_name", "LexVerse"),
                dialog_history_path=combined_path,
                output_dir=output_dir,
                config=eval_config,
                timeout_sec=context.get("timeout_sec"),
            )
        finally:
            combined_path.unlink(missing_ok=True)
        return self._case_results(tasks, evaluated, context), evaluated.aggregate

    def _case_results(self, tasks, evaluated, context, *, include_final=True):
        by_case = {str(item.get("case_id")): item for item in evaluated.per_case}
        results = {}
        for task in tasks:
            case_id = task.source.sample_id
            raw_case = by_case.get(case_id, {})
            raw = {
                "per_case": raw_case,
                "aggregate": {
                    key: evaluated.aggregate[key]
                    for key in ("status", "coverage")
                    if key in evaluated.aggregate
                },
            }
            case_path = evaluated.intermediate_dir / f"{case_id}.json"
            complete = _complete_case(raw_case)
            artifacts = {}
            if include_final and evaluated.final_json.is_file():
                artifacts["final_json"] = str(evaluated.final_json)
            if case_path.is_file():
                artifacts["per_case"] = str(case_path)
            artifacts.update(raw_case.get("native_artifacts", {}))
            results[case_id] = VerifierResult(
                status="completed" if complete else "failed",
                metrics=_metrics(raw_case, task.evaluation.metrics) if complete else {},
                raw_result=raw,
                artifacts=artifacts,
                provenance=VerifierProvenance(
                    name=self.name,
                    version="5.0",
                    upstream_commit=task.source.version,
                    model=context.get("scoring_model"),
                ),
            )
        return results

    def build_groups(self, tasks, results):
        records = [
            {
                "id": task.id,
                "task": task.source.task_type,
                "status": "completed" if task.id in results else "missing",
                "evaluation": results[task.id].model_dump(mode="json") if task.id in results else {},
            }
            for task in tasks
        ]
        aggregated = mean_metrics(records)
        groups = []
        for task_type, value in aggregated.items():
            expected = [task.id for task in tasks if task.source.task_type == task_type]
            scored = [sid for sid in expected if sid in results and results[sid].status == "completed"]
            groups.append({
                "task_type": task_type,
                "task_ids": scored,
                "expected_task_ids": expected,
                "metrics": value["metrics"] if len(scored) == len(expected) else {},
            })
        return groups


class MILEAggregator(BenchmarkAggregator):
    def aggregate(self, records: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
        scored = [
            {
                **record,
                "evaluation": record.get("evaluation") or {"status": "pending", "metrics": {}},
            }
            for record in records
        ]
        tasks = mean_metrics(scored)
        for task_type, group in tasks.items():
            if group["sample_count"] != group["expected_samples"]:
                group["metrics"] = {}
        return {"benchmark": "maser", "model": model_name, "tasks": tasks}

    def write_csv(self, summary: dict[str, Any], path: Path, *, model_name: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["scenario", "model", "metric", "score", "case_count"])
            for scenario, result in summary["tasks"].items():
                for metric, score in result["metrics"].items():
                    writer.writerow([
                        scenario, model_name, metric, f"{score:.6f}",
                        result["sample_count"],
                    ])
        return path


MASEREvalConfig = MILEEvalConfig
MASEREvalResult = MILEEvalResult
MASEREvaluator = MILEEvaluator
MASERTaskVerifier = MILETaskVerifier
MASERAggregator = MILEAggregator
