from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from lexverse.benchmarks.registry import get_plugin
from lexverse.benchmarks.lexeval.aggregate import write_scores_csv as write_lexeval_csv
from lexverse.benchmarks.lawbench.aggregate import write_scores_csv as write_lawbench_csv
from lexverse.config import ConfigError, load_config, load_profile, parse_model_config
from lexverse.providers import OpenAICompatibleProvider
from lexverse.runtime.artifacts import atomic_write_json
from lexverse.runtime.orchestrator import JobSpec, Orchestrator
from lexverse.runtime.reporting import finalize_run
from lexverse.runtime.task_runner import TaskRunner
from lexverse.runtime.paths import trial_relative_path
from lexverse.tasks.bundle import TaskBundle


def cmd_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plugin = get_plugin(config.benchmark["name"])
    bundle = plugin.create_bundle(config)
    output = Path(args.output) if args.output else config.source_path.with_suffix(".prepared.json")
    manifest = {
        "created_at": time.time(), "config_hash": config.hash(),
        "config_source": os.path.relpath(config.source_path, Path.cwd()),
        "raw": config.raw,
        "architecture_version": "1.3",
        "plugin": {
            "name": plugin.name, "source_version": plugin.source_version,
            "default_policy": plugin.default_policy,
            "default_environment": plugin.default_environment,
        },
        "bundle": bundle.model_dump(mode="json"),
    }
    atomic_write_json(output, manifest)
    print(f"[prepare] {len(bundle.tasks)} tasks, bundle={bundle.content_hash()[:16]}, "
          f"config={config.hash()}, wrote {output}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.prepared)
    if not manifest_path.exists():
        print(f"[run] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = load_config(Path(manifest["config_source"]))
    if config.hash() != manifest["config_hash"]:
        print(f"[run] config drift detected: hash {config.hash()} != "
              f"prepared {manifest['config_hash']}; re-run `prepare`", file=sys.stderr)
        return 3
    run_root = Path(args.output_root) if args.output_root else (
        Path("runs") / f"{config.benchmark['name']}-{config.hash()}-{int(time.time())}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "manifest.json", manifest)
    bundle = TaskBundle.model_validate(manifest["bundle"])
    plugin = get_plugin(bundle.benchmark)
    connection = load_profile(config.model.profile)
    if connection.provider != "openai_compatible":
        raise ConfigError(f"unsupported provider: {connection.provider!r}")
    provider = OpenAICompatibleProvider(
        name=connection.name,
        model=config.model.name,
        api_key=connection.api_key,
        base_url=connection.base_url,
        default_config=config.model.parameters,
    )
    environment = plugin.create_environment(config, provider)
    verifier = plugin.create_verifier({
        **config.benchmark,
        "offline": bool(config.execution.get("offline", False)),
    })
    task_runner = TaskRunner()
    tasks = {task.id: task for task in bundle.tasks}

    verifier_context: dict = {}
    if bundle.benchmark == "j1bench":
        scoring_model = parse_model_config(
            config.evaluation.get("model"), field="evaluation.model"
        )
        scoring = load_profile(scoring_model.profile)
        verifier_context = {
            "scoring_api_key": scoring.api_key,
            "scoring_api_base": scoring.base_url,
            "scoring_model": scoring_model.name,
            "case_database": config.benchmark.get("case_database"),
            "case_databases": config.benchmark.get("case_databases"),
            "model_name": config.benchmark.get("model_tag", config.model.name),
            "offline": bool(config.execution.get("offline", False)),
            "timeout_sec": config.evaluation.get("timeout_sec", 1800.0),
        }

    async def execute() -> int:
        await environment.prepare({"run_root": str(run_root)})

        async def factory(task_id: str, work_dir: Path) -> dict:
            task = tasks[task_id]
            environment_result = await task_runner.run_environment(
                task=task, policy=None, participants=None,
                work_dir=work_dir, environment=environment,
            )
            environment_data = environment_result.model_dump(mode="json")
            if bundle.benchmark == "j1bench":
                environment_data["artifacts"]["dialog_history"] = "dialog_history.jsonl"
            return {"trial_result": {
                "id": task.id,
                "benchmark": bundle.benchmark,
                "task": task.source.task_type or bundle.benchmark,
                "sample_id": task.source.sample_id,
                "input": task.input.model_dump(mode="json"),
                "reference": task.evaluation.reference,
                "response": environment_result.answer,
                "interaction": {
                    "messages": environment_data["messages"],
                    "dialog_history": environment_data["dialog_history"],
                    "final_state": environment_data["final_state"],
                    "trace": environment_data["trace"],
                },
                "model": {"name": config.model.name},
                "artifacts": environment_data["artifacts"],
            }}

        job = JobSpec(
            benchmark=bundle.benchmark,
            sample_ids=list(tasks),
            run_dir=run_root,
            resolved_config={
                "model": config.model.name,
                "profile": config.model.profile,
                "parameters": config.model.parameters,
                "bundle_hash": bundle.content_hash(),
            },
            max_attempts=int(config.execution.get("max_attempts", 1)),
            config_hash=manifest["config_hash"],
            trial_paths={
                task.id: str(trial_relative_path(
                    task.source.task_type or bundle.benchmark,
                    task.source.sample_id,
                ))
                for task in bundle.tasks
            },
        )
        try:
            await Orchestrator(
                n_concurrent=int(config.execution.get("concurrency", 1))
            ).run(job, factory, resume=bool(config.execution.get("resume", False)))
            task_metrics: dict[str, dict[str, float]] | None = None
            if bundle.benchmark in {"lexeval", "lawbench"}:
                completed_cases = []
                for task in bundle.tasks:
                    trial_dir = run_root / "trials" / trial_relative_path(
                        task.source.task_type or bundle.benchmark,
                        task.source.sample_id,
                    )
                    trial_data = _read_json(trial_dir / "results.json")
                    if trial_data.get("status") == "completed" and isinstance(
                        trial_data.get("response"), str
                    ):
                        completed_cases.append((task, trial_data["response"]))
                scores = verifier.verify_run(
                    completed_cases,
                    predictions_root=run_root / "verifier" / "predictions",
                    model_name=config.model.name,
                )
                if bundle.benchmark == "lexeval":
                    write_lexeval_csv(
                        scores, run_root / "evaluation_result.csv",
                        model_name=config.model.name,
                    )
                    task_metrics = {
                        score.task_id: {score.metric: score.score} for score in scores
                    }
                else:
                    write_lawbench_csv(
                        scores, run_root / "evaluation_result.csv",
                        model_name=config.model.name,
                    )
                    task_metrics = {
                        score.task_id: {
                            "score": score.score,
                            "abstention_rate": score.abstention_rate,
                        }
                        for score in scores
                    }
            elif bundle.benchmark == "j1bench":
                grouped: dict[str, list] = defaultdict(list)
                for task in bundle.tasks:
                    grouped[str(task.source.task_type)].append(task)
                for scenario, scenario_tasks in grouped.items():
                    completed_tasks = []
                    dialog_paths = []
                    for task in scenario_tasks:
                        trial_dir = (
                            run_root / "trials" / scenario / str(task.source.sample_id)
                        )
                        dialog_path = trial_dir / "dialog_history.jsonl"
                        if dialog_path.is_file():
                            completed_tasks.append(task)
                            dialog_paths.append(dialog_path)
                    if not completed_tasks:
                        continue
                    scenario_dir = run_root / "trials" / scenario
                    verified, _ = await verifier.verify_scenario(
                        completed_tasks, dialog_paths,
                        output_dir=scenario_dir / "verifier",
                        context=verifier_context,
                    )
                    for task in completed_tasks:
                        trial_dir = scenario_dir / str(task.source.sample_id)
                        result_path = trial_dir / "results.json"
                        trial_data = json.loads(result_path.read_text(encoding="utf-8"))
                        verifier_data = verified[task.source.sample_id].model_dump(mode="json")
                        final_path = Path(verifier_data["artifacts"]["final_json"])
                        verifier_data["artifacts"]["final_json"] = os.path.relpath(
                            final_path, run_root
                        )
                        trial_data["evaluation"] = verifier_data
                        trial_data["artifacts"].update(verifier_data["artifacts"])
                        if verifier_data["status"] != "completed":
                            trial_data["status"] = "failed"
                            trial_data["error"] = "official evaluator returned no case result"
                        atomic_write_json(result_path, trial_data)
        finally:
            await environment.close()
        job_result = finalize_run(
            run_root, manifest, plugin, model_name=config.model.name,
            task_metrics=task_metrics,
            write_evaluation_csv=bundle.benchmark == "j1bench",
        )
        return 0 if job_result["status"] == "completed" else 4

    import asyncio
    return asyncio.run(execute())


def cmd_collect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[collect] no manifest.json under {run_dir}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = manifest.get("benchmark") or manifest.get("bundle", {}).get("benchmark")
    if not benchmark:
        print(f"[collect] benchmark missing from {manifest_path}", file=sys.stderr)
        return 2
    plugin = get_plugin(benchmark)
    model_name = (
        manifest.get("model")
        or manifest.get("raw", {}).get("model", {}).get("name")
        or "unknown"
    )
    csv_path = run_dir / "evaluation_result.csv"
    task_metrics = _read_task_metrics(benchmark, csv_path) if csv_path.exists() else None
    output = finalize_run(
        run_dir, manifest, plugin, model_name=model_name,
        task_metrics=task_metrics,
        write_evaluation_csv=not csv_path.exists(),
    )
    samples = output["samples"]
    print(f"[collect] {samples['completed']}/{samples['expected']} trial results")
    return 0 if output["status"] == "completed" else 4


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_task_metrics(benchmark: str, path: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            task = row.get("task") or row.get("scenario")
            if not task:
                continue
            if benchmark == "lexeval":
                metrics.setdefault(task, {})[row["metrics"]] = float(row["score"])
            elif benchmark == "lawbench":
                metrics[task] = {
                    "score": float(row["score"]),
                    "abstention_rate": float(row["abstention_rate"]),
                }
            else:
                metrics.setdefault(task, {})[row["metric"]] = float(row["score"])
    return metrics


def cmd_catalog(args: argparse.Namespace) -> int:
    plugin = get_plugin(args.benchmark)
    print(json.dumps({
        "benchmark": plugin.name, "source_version": plugin.source_version,
        "default_policy": plugin.default_policy,
        "default_environment": plugin.default_environment,
        "tasks": plugin.load_catalog()["tasks"],
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lexverse", description="LexVerse benchmark runner")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="resolve config and freeze the sample manifest")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output")
    prepare.set_defaults(func=cmd_prepare)

    run = commands.add_parser("run", help="execute a prepared benchmark manifest")
    run.add_argument("--prepared", required=True)
    run.add_argument("--output-root")
    run.set_defaults(func=cmd_run)

    collect = commands.add_parser("collect", help="rebuild aggregate results from Trials")
    collect.add_argument("--run-dir", required=True)
    collect.set_defaults(func=cmd_collect)

    catalog = commands.add_parser("catalog", help="inspect a benchmark task catalog")
    catalog.add_argument("benchmark", choices=["lexeval", "lawbench", "j1bench"])
    catalog.set_defaults(func=cmd_catalog)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
