from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from contextlib import contextmanager

from lexverse.benchmarks.registry import available_plugins, get_plugin
from lexverse.config import ConfigError, load_config, parse_config
from lexverse.providers.runtime import ModelRuntime, scoring_identity
from lexverse.runtime.results import atomic_write_json
from lexverse.runtime.orchestrator import JobSpec, Orchestrator
from lexverse.runtime.results import finalize_run
from lexverse.runtime.task_runner import TaskRunner
from lexverse.runtime.errors import TrialTimeoutError
from lexverse.runtime.sources import file_hash as _file_hash, prepare_run_sources, run_identity, source_provenance
from lexverse.runtime.results import trial_relative_path
from lexverse.tasks.bundle import TaskBundle


logger = logging.getLogger("lexverse.run")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _tested_model(config):
    if config.benchmark["name"] in {"dlawbench", "legalworld"}:
        return config.models.roles.get("lawyer", config.models.default)
    return config.models.default


def _model_label(config, label):
    if label.startswith("models.evaluators."):
        if "models" not in config.raw:
            key = "judge"
            if config.benchmark["name"] == "dlawbench" and config.evaluation.get("mode") == "panel":
                index = int(label.removeprefix("models.evaluators."))
                key = config.models.evaluators[index].name
            return f"evaluation.{key}" if config.benchmark["name"] in {"dlawbench", "legalworld"} else "evaluation.model"
        return label
    if "models" not in config.raw:
        return {"models.default": "model", "models.summary": "benchmark.summary_model"}.get(
            label, label.replace("models.roles.", "roles."),
        )
    return label


def _read_task_metrics(path: Path) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            task = row.get("task") or row.get("scenario")
            if not task:
                continue
            if "metrics" in row:
                metrics.setdefault(task, {})[row["metrics"]] = float(row["score"])
            elif "abstention_rate" in row:
                metrics[task] = {
                    "score": float(row["score"]),
                    "abstention_rate": float(row["abstention_rate"]),
                }
            else:
                metrics.setdefault(task, {})[row["metric"]] = float(row["score"])
    return metrics


@contextmanager
def _run_logging(run_root: Path | None):
    """Scope terminal/file handlers to one invocation; resume appends its log."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_root / "run.log", encoding="utf-8"))
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    previous = logger.handlers[:], logger.level, logger.propagate
    for handler in handlers:
        handler.setFormatter(formatter)
    logger.handlers, logger.propagate = handlers, False
    logger.setLevel(logging.INFO)
    try:
        yield
    except BaseException as exc:
        # Detailed exceptions remain in trial/evaluation records; connection
        # credentials or model content must not leak through exception messages.
        logger.error("[run] failed error=%s", type(exc).__name__)
        raise
    finally:
        logger.handlers, level, logger.propagate = previous
        logger.setLevel(level)
        for handler in handlers:
            handler.close()


def build_run(config):
    plugin = get_plugin(config.benchmark["name"])
    bundle = plugin.create_bundle(config)
    counts = plugin.selection_counts
    for kind, count in counts.items():
        requested = "all" if count["requested"] is None else count["requested"]
        logger.info("[run] %s: requested=%s, available=%s, selected=%s",
                    kind, requested, count['available'], count['selected'])
    if plugin.name == "dlawbench":
        from lexverse.benchmarks.dlawbench.tasks import load_cases, PERSONAS
        cases = load_cases(plugin.upstream.ensure(offline=True))
        logger.info("[run] public_cases=%s personas=%s valid_combinations=%s selected=%s",
                    len(cases), len(PERSONAS), len(cases) * len(PERSONAS), len(bundle.tasks))
    if plugin.name == "legalworld":
        from lexverse.benchmarks.legalworld.tasks import LegalWorldImporter
        cases = LegalWorldImporter(offline=True).cases
        ranges = sorted({f"{task.metadata['start_stage']}:{task.metadata['end_stage']}" for task in bundle.tasks})
        roles = sorted({task.metadata["party_role"] for task in bundle.tasks})
        logger.info("[run] public_cases=%s roles=%s ranges=%s valid_combinations=%s selected=%s",
                    len(cases), ",".join(roles), ",".join(ranges),
                    len(cases) * len({(task.metadata["party_role"], task.metadata["start_stage"],
                                      task.metadata["end_stage"]) for task in bundle.tasks}), len(bundle.tasks))
    manifest = {
        "snapshot_version": 1.0, "created_at": time.time(), "config_hash": config.hash(),
        "config_source": str(config.source_path), "architecture_version": "1.0",
        "config": config.raw,
        "selected_tasks": [{"id": task.id, "task_type": task.source.task_type,
                            "sample_id": task.source.sample_id} for task in bundle.tasks],
        "data_sources": [{"name": name, "path": str(path.resolve()), "sha256": _file_hash(path)}
                         for name, path in plugin.task_inputs(config, list(counts)).items()],
        "benchmark": bundle.benchmark, "model": _tested_model(config).name,
        "sample_count": len(bundle.tasks), "selection": counts,
        "bundle_hash": bundle.content_hash(),
        "coverage": {
            "scope": "full" if set(counts) == (set(entry["id"] for entry in config.benchmark["tasks"])
                                              if plugin.name == "legalworld" else set(plugin.task_types()))
            and not config.benchmark.get("cases")
            and all(count["selected"] == count["available"] for count in counts.values()) else "selected",
            "task_types": list(counts), "sample_count": len(bundle.tasks),
        },
        "provenance": source_provenance(
            plugin, offline=bool(config.generation.get("offline", False)), config=config,
        ),
        "bundle": bundle.model_dump(mode="json"),
    }
    return plugin, bundle, manifest


def load_run_snapshot(run_root):
    manifest = _read_json(run_root / "manifest.json")
    if manifest.get("models_sha256"):
        models = run_root / "models.json"
        if not models.is_file() or _file_hash(models) != manifest["models_sha256"]:
            raise ConfigError("model provenance snapshot mismatch")
    version = manifest.get("snapshot_version")
    if version == 1 and "snapshot_hashes" in manifest:
        # Read existing snapshots without converting the user's old run directory.
        for name in ("config.json", "tasks.json"):
            path = run_root / name
            if not path.is_file() or _file_hash(path) != manifest["snapshot_hashes"].get(name):
                raise ConfigError(f"missing or changed run snapshot: {name}")
        config = parse_config(_read_json(run_root / "config.json"), source_path=run_root / "config.json")
        bundle = TaskBundle.model_validate(_read_json(run_root / "tasks.json"))
        plugin = get_plugin(bundle.benchmark)
    elif version in (1.0, 2):
        identity_hash = hashlib.sha256(json.dumps(run_identity(manifest), sort_keys=True).encode()).hexdigest()
        if identity_hash != manifest.get("run_hash"):
            raise ConfigError("run manifest snapshot mismatch")
        config = parse_config(manifest.get("config", {}), source_path=run_root / "manifest.json")
        if config.hash() != manifest.get("config_hash"):
            raise ConfigError("run config snapshot mismatch")
        plugin = get_plugin(config.benchmark["name"])
        for entry in manifest.get("data_sources", []):
            path = Path(entry["path"])
            if not path.is_file() or _file_hash(path) != entry["sha256"]:
                raise ConfigError(f"shared dataset missing or changed: {path}")
        references = manifest.get("selected_tasks")
        if not isinstance(references, list) or not references:
            raise ConfigError("run manifest has no selected task IDs")
        inputs = plugin.task_inputs(config, list(dict.fromkeys(ref["task_type"] for ref in references)))
        expected_hashes = {entry["name"]: entry["sha256"] for entry in manifest.get("data_sources", [])}
        if (set(inputs) != set(expected_hashes)
                or any(not path.is_file() or _file_hash(path) != expected_hashes[name]
                       for name, path in inputs.items())):
            raise ConfigError("shared dataset used by importer has changed")
        try:
            bundle = plugin.restore_bundle(config, references)
        except (ValueError, KeyError) as exc:
            raise ConfigError(f"cannot reconstruct selected tasks: {exc}") from exc
    else:
        raise ConfigError("old run has no task selection; start a new run --config CONFIG")
    if (config.hash() != manifest["config_hash"] or bundle.content_hash() != manifest["bundle_hash"]
            or bundle.benchmark != manifest["benchmark"] or len(bundle.tasks) != manifest["sample_count"]):
        raise ConfigError("run config/task snapshot mismatch")
    manifest["bundle"] = bundle.model_dump(mode="json")
    return config, plugin, bundle, manifest


async def _evaluate_run(config, plugin, bundle, manifest, run_root, runtime, verifier):
    request_timeout = config.evaluation.get("model_request_timeout_sec")
    if runtime.request_timeout != request_timeout:
        runtime.stop_worker()
    runtime.request_timeout = request_timeout
    runtime.startup_timeout = config.evaluation.get("model_startup_timeout_sec", 1800)
    records = {
        task.id: _read_json(run_root / "trials" / trial_relative_path(
            task.source.task_type or bundle.benchmark, task.source.sample_id,
        ) / "results.json") for task in bundle.tasks
    }
    for record in records.values():
        if record.get("error") == "official evaluator returned no case result":
            record["status"] = "completed"
            record.pop("error", None)
    verifier_context = plugin.evaluation_context(config)
    if plugin.name == "j1bench":
        if not config.models.evaluators:
            raise ConfigError("J1Bench requires one model in models.evaluators")
        runtime.validate_phase(config.models.evaluators)
        scoring = runtime.provider(config.models.evaluators[0], label=_model_label(config, "models.evaluators.0"))
        verifier_context.update(
            scoring_api_key=scoring.api_key, scoring_api_base=scoring.base_url,
            scoring_provider=scoring,
            request_timeout_sec=request_timeout,
            scoring_model=scoring.model, scoring_parameters=scoring.default_config,
            python_executable=sys.executable,
        )
    if plugin.name == "plawbench":
        if not config.models.evaluators:
            raise ConfigError("PLawBench requires one model in models.evaluators")
        runtime.validate_phase(config.models.evaluators)
        judge = runtime.provider(config.models.evaluators[0], label=_model_label(config, "models.evaluators.0"))
        verifier_context.update(
            judge_provider=judge,
            judge_model=config.models.evaluators[0].name,
            judge_config={**asdict(config.models.evaluators[0]),
                          "endpoint_sha256": scoring_identity(judge)["endpoint_sha256"]},
        )
    if plugin.name in {"dlawbench", "legalworld"}:
        models = verifier_context.pop("judge_models")
        runtime.validate_phase(list(models.values()))
        verifier_context.update(
            judge_connections={key: runtime.provider(model, label=_model_label(config, f"models.evaluators.{config.models.evaluators.index(model)}"))
                               for key, model in models.items()},
            timeout_sec=config.evaluation.get("timeout_sec"),
        )
    evaluation_started = time.monotonic()
    logger.info("[evaluation] start")
    try:
        verifier_context.update(
            concurrency=int(config.evaluation.get("concurrency", config.generation.get("concurrency", 1))),
            evaluation_config={key: value for key, value in config.evaluation.items() if key != "concurrency"},
            on_progress=lambda result: finalize_run(
                run_root, manifest, plugin, model_name=_tested_model(config).name,
                task_metrics=result.task_metrics(), write_evaluation_csv=False,
            ),
        )
        evaluated = await verifier.evaluate_run(
            bundle.tasks, records, run_dir=run_root,
            model_name=_tested_model(config).name, context=verifier_context,
        )
    except Exception as exc:
        logger.error("[evaluation] failed error=%s elapsed=%.1fs",
                     type(exc).__name__, time.monotonic() - evaluation_started)
        evaluation_state = _read_json(run_root / "evaluation.json")
        evaluation_state.update(status="failed", error_type=type(exc).__name__)
        atomic_write_json(run_root / "evaluation.json", evaluation_state)
        finalize_run(
            run_root, manifest, plugin, model_name=_tested_model(config).name,
            task_metrics={}, write_evaluation_csv=False,
        )
        raise
    evaluation_data = evaluated.to_dict()
    task_metrics = evaluated.task_metrics()
    finalize_run(run_root, manifest, plugin, model_name=_tested_model(config).name,
                 task_metrics=task_metrics, write_evaluation_csv=not getattr(verifier, "grouped", False))
    logger.info("[evaluation] %s elapsed=%.1fs scored=%s/%s",
                evaluation_data["status"], time.monotonic() - evaluation_started,
                len(evaluation_data["scored_task_ids"]), len(bundle.tasks))
    return task_metrics


def cmd_run(args: argparse.Namespace) -> int:
    resume_dir = getattr(args, "resume", None)
    benchmark = getattr(args, "benchmark", None)
    config_path = getattr(args, "config", None)
    if resume_dir and (benchmark or config_path):
        raise ConfigError("--resume cannot be combined with a benchmark name or --config")
    if not resume_dir and not (benchmark or config_path):
        raise ConfigError("provide a benchmark name, --config, or --resume RUN_DIR")
    if resume_dir:
        if getattr(args, "dry_run", False) or getattr(args, "output_root", None):
            raise ConfigError("--dry-run and --output-root require a new run")
        run_root = Path(resume_dir).resolve()
        config, plugin, bundle, manifest = load_run_snapshot(run_root)
    else:
        if not config_path:
            directory = Path("configs/benchmarks")
            candidates = [directory / f"{benchmark}.local.yaml", directory / f"{benchmark}.example.yaml"]
            config_path = next((path for path in candidates if path.is_file()), None)
            if config_path is None:
                raise ConfigError(f"no local/example configuration for {benchmark}; use --config CONFIG")
        config = load_config(config_path)
        if benchmark and config.benchmark["name"] != benchmark:
            raise ConfigError(f"benchmark name {benchmark} does not match configuration {config.benchmark['name']}")
        run_root = Path(args.output_root).resolve() if getattr(args, "output_root", None) else (
            Path("runs") / f"{config.benchmark['name']}-{config.hash()}-{time.time_ns()}"
        ).resolve()
        if not getattr(args, "dry_run", False) and run_root.exists() and any(run_root.iterdir()):
            raise ConfigError("run directory is not empty; use --resume RUN_DIR or a new directory")
    dry_run = bool(getattr(args, "dry_run", False))
    with _run_logging(None if dry_run else run_root):
        run_started = time.monotonic()
        logger.info("[run] config=%s", config.source_path)
        logger.info("[run] preparing benchmark=%s resume=%s", config.benchmark["name"], bool(resume_dir))
        if not resume_dir:
            plugin, bundle, manifest = build_run(config)
            if getattr(args, "dry_run", False):
                logger.info("[dry-run] %s tasks, source=%s, patchset=%s",
                            len(bundle.tasks), bundle.source_version, manifest["provenance"]["patchset_hash"])
                return 0
        logger.info("[run] preparing sources")
        run_hash = prepare_run_sources(
            run_root, manifest, plugin, resume=bool(resume_dir),
            offline=bool(config.generation.get("offline", False)), config=config,
        )
        manifest["run_hash"] = run_hash
        atomic_write_json(run_root / "manifest.json", {key: value for key, value in manifest.items() if key != "bundle"})
        logger.info("[run] directory=%s", run_root)
        task_runner = TaskRunner()
        tasks = {task.id: task for task in bundle.tasks}
        runtime = ModelRuntime(
            run_root, startup_timeout=config.generation.get("model_startup_timeout_sec", 1800),
            request_timeout=config.generation.get("model_request_timeout_sec"),
        )
        runtime.bind_manifest(manifest)

        async def execute() -> int:
            if plugin.name == "plawbench":
                if not config.models.evaluators:
                    raise ConfigError("PLawBench requires one model in models.evaluators")
                plugin.evaluation_context(config)
            if plugin.name == "j1bench":
                if not config.models.evaluators:
                    raise ConfigError("J1Bench requires one model in models.evaluators")
                from lexverse.benchmarks.j1bench.scenarios import require_supported
                role_names = dict.fromkeys(role for task in bundle.tasks for role in require_supported(task.source.task_type).roles)
                models = {role: config.models.roles.get(role, config.models.default) for role in role_names}
                uses_summary = any(task.source.task_type in {"KQ", "LC"} for task in bundle.tasks)
                if uses_summary:
                    models["summary"] = config.models.summary
                runtime.validate_phase(list(models.values()))
                connections = {role: runtime.provider(model, label=_model_label(config,
                    "models.summary" if role == "summary" else f"models.roles.{role}"
                )) for role, model in models.items()}
                environment = plugin.create_environment(config, None)
                environment.connections = connections
            elif plugin.name in {"dlawbench", "legalworld"}:
                models = {role: config.models.roles.get(role, config.models.default) for role in (("lawyer", "client") if plugin.name == "dlawbench" else ("lawyer", "simulation"))}
                runtime.validate_phase(list(models.values()))
                environment = plugin.create_environment(config, None)
                environment.connections = {role: runtime.provider(model, label=_model_label(config, f"models.roles.{role}"))
                                           for role, model in models.items()}
            else:
                runtime.validate_phase([config.models.default])
                environment = plugin.create_environment(config, runtime.provider(config.models.default, label=_model_label(config, "models.default")))
            verifier = plugin.create_verifier({**config.benchmark, "offline": bool(config.generation.get("offline", False))})
            await environment.prepare({"run_root": str(run_root)})
            logger.info("[run] executing total=%s concurrency=%s",
                        len(tasks), config.generation.get("concurrency", 1))

            async def factory(task_id: str, work_dir: Path) -> dict:
                task = tasks[task_id]
                try:
                    environment_result = await asyncio.wait_for(task_runner.run_environment(
                        task=task, policy=None, participants=None,
                        work_dir=work_dir, environment=environment,
                    ), timeout=config.generation.get("timeout_sec"))
                except TrialTimeoutError:
                    raise
                except asyncio.TimeoutError as exc:
                    if config.generation.get("timeout_sec") is None:
                        raise
                    raise TrialTimeoutError(f"task execution exceeded {config.generation.get('timeout_sec')}s") from exc
                environment_data = environment_result.model_dump(mode="json")
                environment_data["artifacts"] = {
                    name: os.path.relpath(path, run_root) if Path(path).is_absolute() else path
                    for name, path in environment_data["artifacts"].items()
                }
                return {"trial_result": {
                    "id": task.id,
                    "benchmark": bundle.benchmark,
                    "source": task.source.model_dump(mode="json"),
                    "run_hash": run_hash,
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
                    "model": {"name": _tested_model(config).name},
                    "artifacts": environment_data["artifacts"],
                    "execution_artifacts": dict(environment_data["artifacts"]),
                }}

            job = JobSpec(
                benchmark=bundle.benchmark,
                sample_ids=list(tasks),
                run_dir=run_root,
                resolved_config={
                    "model": _tested_model(config).name,
                    "provider": _tested_model(config).provider,
                    "generation_parameters": _tested_model(config).generation_parameters,
                    "bundle_hash": bundle.content_hash(),
                },
                max_attempts=int(config.generation.get("max_attempts", 1)),
                config_hash=manifest["config_hash"],
                run_hash=run_hash,
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
                    n_concurrent=int(config.generation.get("concurrency", 1))
                ).run(job, factory, resume=bool(resume_dir))
                task_metrics = await _evaluate_run(config, plugin, bundle, manifest, run_root, runtime, verifier)
            finally:
                await environment.close()
            job_result = finalize_run(
                run_root, manifest, plugin, model_name=_tested_model(config).name,
                task_metrics=task_metrics,
                write_evaluation_csv=not (run_root / "evaluation_result.csv").exists(),
            )
            logger.info("[run] %s success=%s failed=%s elapsed=%.1fs directory=%s",
                        job_result["status"], job_result["samples"]["completed"],
                        job_result["samples"]["failed"], time.monotonic() - run_started, run_root)
            return 0 if job_result["status"] == "completed" else 4

        import asyncio
        async def managed_execute():
            try:
                return await execute()
            finally:
                await runtime.close()
        return asyncio.run(managed_execute())


def cmd_evaluate(args: argparse.Namespace) -> int:
    import asyncio
    run_root = Path(args.run_dir).resolve()
    config, plugin, bundle, manifest = load_run_snapshot(run_root)
    prepare_run_sources(run_root, manifest, plugin, resume=True,
                        offline=bool(config.generation.get("offline", False)), config=config)
    with _run_logging(run_root):
        runtime = ModelRuntime(run_root,
                               startup_timeout=config.evaluation.get("model_startup_timeout_sec", 1800),
                               request_timeout=config.evaluation.get("model_request_timeout_sec"))
        runtime.bind_manifest(manifest)
        verifier = plugin.create_verifier({**config.benchmark, "offline": bool(config.generation.get("offline", False))})
        async def execute():
            try:
                metrics = await _evaluate_run(config, plugin, bundle, manifest, run_root, runtime, verifier)
                result = finalize_run(run_root, manifest, plugin, model_name=_tested_model(config).name,
                                      task_metrics=metrics, write_evaluation_csv=False)
                return 0 if result["status"] == "completed" else 4
            finally:
                await runtime.close()
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
    if manifest.get("snapshot_version") in (1, 2):
        _, _, bundle, manifest = load_run_snapshot(run_dir)
    plugin = get_plugin(benchmark)
    model_name = (
        manifest.get("model")
        or manifest.get("raw", {}).get("model", {}).get("name")
        or "unknown"
    )
    csv_path = run_dir / "evaluation_result.csv"
    evaluation = _read_json(run_dir / "evaluation.json")
    task_metrics = (
        {group["task_type"]: group["metrics"] for group in evaluation["groups"]}
        if "groups" in evaluation else _read_task_metrics(csv_path) if csv_path.exists() else None
    )
    output = finalize_run(
        run_dir, manifest, plugin, model_name=model_name,
        task_metrics=task_metrics,
        write_evaluation_csv=not csv_path.exists(),
    )
    samples = output["samples"]
    print(f"[collect] {samples['completed']}/{samples['expected']} trial results")
    return 0 if output["status"] == "completed" else 4


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
    benchmark_names = available_plugins()

    run = commands.add_parser("run", help="prepare and run a benchmark, or resume its frozen tasks")
    run.add_argument("benchmark", nargs="?", type=str.lower, choices=benchmark_names)
    inputs = run.add_mutually_exclusive_group()
    inputs.add_argument("--config")
    inputs.add_argument("--resume", metavar="RUN_DIR")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--output-root")
    run.set_defaults(func=cmd_run)

    evaluate = commands.add_parser("evaluate", help="rerun native scoring using the frozen run configuration")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    collect = commands.add_parser("collect", help="rebuild aggregate results from Trials")
    collect.add_argument("--run-dir", required=True)
    collect.set_defaults(func=cmd_collect)

    catalog = commands.add_parser("catalog", help="inspect a benchmark task catalog")
    catalog.add_argument("benchmark", choices=benchmark_names)
    catalog.set_defaults(func=cmd_catalog)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
