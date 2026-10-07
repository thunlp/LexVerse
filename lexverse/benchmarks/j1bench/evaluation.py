from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from lexverse.verifiers.base import (
    TaskVerifier, BenchmarkAggregator,
    mean_metrics,
    RunEvaluationResult,
    VerifierProvenance,
    VerifierResult,
)
from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.executor import run_subprocess
from lexverse.runtime.results import artifact_model_name
from lexverse.runtime.sources import UpstreamSource
from lexverse.benchmarks.j1bench import UPSTREAM as J1BENCH
from lexverse.tasks.schema import LexVerseTask

from .scenarios import require_supported
from .tasks import resolve_case_database


@dataclass
class J1BenchEvalConfig:
    scoring_api_key: str
    scoring_api_base: str | None = None
    scoring_model: str = "gpt-4o-mini"
    scoring_parameters: dict[str, Any] = field(default_factory=dict)
    request_timeout_sec: float | None = None


@dataclass
class J1BenchEvalResult:
    scenario: str
    model_name: str
    intermediate_dir: Path
    final_json: Path
    per_case: list[dict]
    aggregate: dict


def _case_id(value, path):
    case_id = str(value.get("case_id", path.stem))
    scenario = path.parent.name
    suffix = case_id.removeprefix(f"{scenario}_")
    return f"{scenario}-{suffix}" if suffix.isdigit() else case_id


def _metrics(aggregate: dict, expected: list[str]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in aggregate.items():
        if isinstance(value, (int, float)):
            flattened[key] = float(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (int, float)):
                    flattened[nested_key] = float(nested_value)
    return {name: flattened[name] for name in expected if name in flattened}


def _prepare_intermediate(directory: Path, tasks, *, resume,
                          resume_task_ids=None, artifact_hashes=None):
    """Keep readable native cases with matching inputs and saved artifact hashes."""
    expected = {task.source.sample_id for task in tasks
                if resume_task_ids is None or task.id in resume_task_ids}
    for path in directory.glob("*.json"):
        try:
            value = json.loads(path.read_text())
            valid = ((resume or resume_task_ids is not None) and isinstance(value, dict) and bool(value)
                     and _case_id(value, path) in expected
                     and _case_id(value, path) == _case_id({}, path))
            # Python's JSON reader accepts NaN/Infinity; native results must not.
            json.dumps(value, allow_nan=False)
            saved_hash = (artifact_hashes or {}).get(str(path.resolve()))
            if saved_hash is not None:
                valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == saved_hash
        except (OSError, ValueError, TypeError):
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
            if isinstance(value, dict):
                # Native scorers use numeric IDs or CD_1 filenames, while tasks use CD-1.
                value["case_id"] = _case_id(value, path)
                out.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    return out


class J1BenchEvaluator:
    def __init__(self, scenario: str, upstream: UpstreamSource = J1BENCH,
                 offline: bool = False, python_executable: str = "python",
                 case_database: Path | None = None):
        self.scenario = require_supported(scenario).name
        self._python = python_executable
        self._cache = upstream.ensure_patched(
            Path(__file__).parent / "patches", offline=offline
        )
        self._eval_py = (
            self._cache / "src" / "Eval" / "bench" / self.scenario
            / f"{self.scenario}.py"
        )
        if not self._eval_py.exists():
            raise EvaluationError(
                f"upstream missing evaluator for {self.scenario}: {self._eval_py}"
            )
        if case_database is None or not Path(case_database).is_file():
            raise EvaluationError(
                f"download J1-Eval_{self.scenario}.jsonl from "
                "https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset/tree/main "
                "and pass case_database explicitly"
            )
        self._case_database = Path(case_database).resolve()

    async def evaluate(self, model_name: str, dialog_history_path: Path,
                       output_dir: Path, config: J1BenchEvalConfig, *,
                       timeout_sec: float | None = None) -> J1BenchEvalResult:
        model_tag = artifact_model_name(model_name)
        output_dir = output_dir.resolve()
        dialog_history_path = dialog_history_path.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        dh_root = output_dir / "dialog_history"
        int_root = output_dir / "intermediate"
        fin_root = output_dir / "final"
        (dh_root / model_tag).mkdir(parents=True, exist_ok=True)
        int_root.mkdir(parents=True, exist_ok=True)
        fin_root.mkdir(parents=True, exist_ok=True)
        dest = dh_root / model_tag / f"{self.scenario}_dialog_history.jsonl"
        shutil.copyfile(dialog_history_path, dest)

        env = dict(os.environ)
        env["J1BENCH_ROOT"] = str(self._cache)
        env["J1BENCH_CASE_DATABASE"] = str(self._case_database)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self._cache / "src"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env["OPENAI_API_KEY"] = config.scoring_api_key
        env.pop("OPENAI_API_BASE", None)
        if config.scoring_api_base:
            env["OPENAI_API_BASE"] = config.scoring_api_base
        env["J1BENCH_SCORING_MODEL"] = config.scoring_model
        env["J1BENCH_SCORING_PARAMETERS"] = json.dumps(config.scoring_parameters)
        env.pop("J1BENCH_SCORING_REQUEST_TIMEOUT", None)
        if config.request_timeout_sec is not None:
            env["J1BENCH_SCORING_REQUEST_TIMEOUT"] = str(config.request_timeout_sec)
        await run_subprocess(
            command=[self._python, str(self._eval_py),
                     "--dialog_history_dir", dh_root.name,
                     "--intermediate_eval", int_root.name,
                     "--final_eval", fin_root.name],
            work_dir=output_dir, timeout_sec=timeout_sec, env=env,
        )

        intermediate = int_root / model_tag / self.scenario
        per_case = _read_per_case(intermediate)
        # Every upstream evaluator writes <model>_final.json.
        final_json = fin_root / f"{model_tag}_final.json"
        if not final_json.is_file():
            raise EvaluationError(
                f"J1Bench {self.scenario} evaluator produced no final result: "
                f"{final_json}"
            )
        aggregate = json.loads(final_json.read_text(encoding="utf-8"))
        return J1BenchEvalResult(self.scenario, model_name, intermediate,
                                 final_json, per_case, aggregate)


class J1BenchTaskVerifier(TaskVerifier):
    name = "j1bench"
    grouped = True

    def prepare_unit(self, tasks, records, *, run_dir, model_name, context):
        scenario = tasks[0].source.task_type
        directory = run_dir / "trials" / scenario / "verifier" / "intermediate" / artifact_model_name(model_name) / scenario
        eligible = [task for task in tasks if records.get(task.id, {}).get("status") == "completed"]
        _prepare_intermediate(directory, eligible, resume=context.get("resume_native", False),
                              resume_task_ids=context.get("resume_task_ids"),
                              artifact_hashes=context.get("resume_artifact_hashes"))

    async def evaluate_unit(self, tasks, records, *, run_dir, model_name, context):
        from collections import defaultdict

        grouped = defaultdict(list)
        for task in tasks:
            grouped[task.source.task_type].append(task)
        result = RunEvaluationResult(expected_task_ids=[task.id for task in tasks])
        for scenario, scenario_tasks in grouped.items():
            completed_tasks, dialog_paths = [], []
            for task in scenario_tasks:
                record = records.get(task.id, {})
                dialog = (record.get("artifacts") or {}).get("dialog_history")
                path = run_dir / dialog if dialog else None
                if record.get("status") == "completed" and path is not None and path.is_file():
                    completed_tasks.append(task)
                    dialog_paths.append(path)
                elif record.get("status") == "completed":
                    result.trials[task.id] = VerifierResult(
                        status="invalid", provenance=VerifierProvenance(
                            name=self.name, version="1.0", upstream_commit=task.source.version,
                        ),
                    )
            if not completed_tasks:
                continue
            verified, aggregate = await self.verify_scenario(
                completed_tasks, dialog_paths,
                output_dir=run_dir / "trials" / scenario / "verifier",
                context={"model_name": model_name, **(context or {})},
            )
            for task in completed_tasks:
                result.trials[task.id] = verified[task.source.sample_id]
            result.groups.append({
                "task_type": scenario,
                "granularity": "scenario",
                "task_ids": [task.id for task in completed_tasks],
                "expected_task_ids": [task.id for task in scenario_tasks],
                "metrics": _metrics(aggregate, scenario_tasks[0].evaluation.metrics),
                "raw_result": aggregate,
                "artifacts": {
                    name: str(Path(path).relative_to(run_dir.resolve())) if Path(path).is_absolute() else path
                    for name, path in verified[completed_tasks[0].source.sample_id].artifacts.items()
                },
                "provenance": verified[completed_tasks[0].source.sample_id].provenance.model_dump(mode="json"),
            })
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
        aggregator = J1BenchAggregator()
        summary = aggregator.aggregate(scored_records, model_name=model_name)
        for group in result.groups:
            summary["tasks"][group["task_type"]]["metrics"] = group["metrics"]
        aggregator.write_csv(summary, run_dir / "evaluation_result.csv", model_name=model_name)

    async def verify_scenario(
        self,
        tasks: list[LexVerseTask],
        dialog_paths: list[Path],
        *,
        output_dir: Path,
        context: dict | None = None,
    ) -> tuple[dict[str, VerifierResult], dict]:
        """Run the upstream evaluator once over every case in one scenario."""
        if not tasks or len(tasks) != len(dialog_paths):
            raise ValueError("J1Bench scenario verification needs matching tasks and dialogs")
        context = context or {}
        scenario = tasks[0].source.task_type
        if any(task.source.task_type != scenario for task in tasks):
            raise ValueError("J1Bench scenario verification cannot mix scenarios")
        api_key = context.get("scoring_api_key")
        if not api_key:
            raise ValueError("J1Bench verification requires context.scoring_api_key")

        configured_paths = context.get("case_databases") or {}
        configured_path = configured_paths.get(scenario) or context.get("case_database")
        case_database = (
            Path(configured_path)
            if configured_path
            else resolve_case_database(scenario, offline=bool(context.get("offline", False)))
        )
        evaluator = J1BenchEvaluator(
            scenario,
            offline=context.get("offline", False),
            python_executable=context.get("python_executable", "python"),
            case_database=case_database,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate = output_dir / "intermediate" / artifact_model_name(context.get("model_name", "LexVerse")) / scenario
        _prepare_intermediate(intermediate, tasks, resume=context.get("resume_native", False),
                              resume_task_ids=context.get("resume_task_ids"),
                              artifact_hashes=context.get("resume_artifact_hashes"))
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False,
            dir=output_dir,
        ) as combined:
            combined_path = Path(combined.name)
            for path in dialog_paths:
                text = path.read_text(encoding="utf-8")
                combined.write(text)
                if text and not text.endswith("\n"):
                    combined.write("\n")
        try:
            evaluated = await evaluator.evaluate(
                model_name=context.get("model_name", "LexVerse"),
                dialog_history_path=combined_path,
                output_dir=output_dir,
                config=J1BenchEvalConfig(
                    scoring_api_key=api_key,
                    scoring_api_base=context.get("scoring_api_base"),
                    scoring_model=context.get("scoring_model", "gpt-4o-mini"),
                    scoring_parameters=context.get("scoring_parameters", {}),
                    request_timeout_sec=context.get("request_timeout_sec"),
                ),
                timeout_sec=context.get("timeout_sec"),
            )
        finally:
            combined_path.unlink(missing_ok=True)

        by_case = {
            str(item.get("case_id")): item
            for item in evaluated.per_case
            if isinstance(item, dict) and item.get("case_id")
        }
        case_paths = {_case_id(json.loads(path.read_text()), path): path
                      for path in evaluated.intermediate_dir.glob("*.json")}
        results: dict[str, VerifierResult] = {}
        for task in tasks:
            case_id = task.source.sample_id
            raw_case = by_case.get(case_id)
            results[case_id] = VerifierResult(
                status="completed" if raw_case is not None else "invalid",
                metrics=_metrics(raw_case or {}, task.evaluation.metrics),
                raw_result={"per_case": raw_case or {}, "aggregate": evaluated.aggregate},
                artifacts={"final_json": str(evaluated.final_json), **(
                    {"per_case": str(case_paths[case_id])} if case_id in case_paths else {})},
                provenance=VerifierProvenance(
                    name=self.name,
                    version="1.0",
                    upstream_commit=task.source.version,
                    model=context.get("scoring_model"),
                ),
            )
        return results, evaluated.aggregate


class J1BenchAggregator(BenchmarkAggregator):
    def aggregate(self, records: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
        tasks = mean_metrics(records)
        # The upstream evaluator's scenario-level final JSON is authoritative.
        # Per-case means are only a fallback for older runs.
        for record in records:
            scenario = str(record.get("task", ""))
            aggregate = ((record.get("evaluation") or {}).get("raw_result") or {}).get(
                "aggregate"
            )
            if scenario in tasks and isinstance(aggregate, dict):
                expected = set(tasks[scenario]["metrics"])
                flattened: dict[str, float] = {}
                for key, value in aggregate.items():
                    if isinstance(value, (int, float)):
                        flattened[key] = float(value)
                    elif isinstance(value, dict):
                        flattened.update({
                            nested_key: float(nested_value)
                            for nested_key, nested_value in value.items()
                            if isinstance(nested_value, (int, float))
                        })
                tasks[scenario]["metrics"] = {
                    key: flattened[key] for key in expected if key in flattened
                }
        return {"benchmark": "j1bench", "model": model_name, "tasks": tasks}

    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path:
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
