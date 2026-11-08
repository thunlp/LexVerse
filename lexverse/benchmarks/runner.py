from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
from dataclasses import asdict
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from lexverse.config import ConfigError, parse_config, resolve_benchmark_config
from lexverse.llms.runtime import ModelRuntime, scoring_identity
from lexverse.benchmarks.registry import get_plugin
from lexverse.benchmarks.sources import file_hash as _file_hash, run_identity, source_provenance
from lexverse.job import Job, export_excel, run_job, summarize, write_summary, write_verification_report, load_verification, save_verification
from lexverse.models.trial.config import ExecutionConfig
from lexverse.trial.trial import Trial
from lexverse.utils.identity import identity
from lexverse.models.trial.paths import new_run_id
from lexverse.utils.files import atomic_write_json
from lexverse.tasks.bundle import TaskBundle


logger = logging.getLogger("lexverse.run")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    value = json.loads(path.read_text(encoding="utf-8"))

    return value if isinstance(value, dict) else {}


def _tested_model(config):
    return get_plugin(config.benchmark["name"]).tested_model(config)


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
        "snapshot_version": "1.0", "created_at": time.time(), "config_hash": config.hash(),
        "config_source": str(config.source_path), "architecture_version": "1.0",
        "data_sources": [{"name": name, "path": str(path.resolve()), "sha256": _file_hash(path)}
                         for name, path in plugin.task_inputs(config, list(counts)).items()],
        "benchmark": bundle.benchmark, "model": _tested_model(config).name,
        "sample_count": len(bundle.tasks), "selection": counts,
        "bundle_hash": bundle.content_hash(legacy=config.legacy),
        "coverage": {
            "scope": "full" if set(counts) == (set(entry["id"] for entry in config.benchmark["tasks"])
                                              if plugin.name == "legalworld" else set(plugin.task_types()))
            and not config.benchmark.get("cases")
            and not any(isinstance(entry, dict) and entry.get("cases") for entry in config.benchmark.get("tasks") or [])
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

    if manifest.get("snapshot_version") != "1.0":
        raise ConfigError("Unsupported source snapshot; start a new Job")

    if manifest.get("models_sha256"):
        models = run_root / "models.json"
        if not models.is_file() or _file_hash(models) != manifest["models_sha256"]:
            raise ConfigError("model provenance snapshot mismatch")

    identity_hash = hashlib.sha256(json.dumps(run_identity(manifest), sort_keys=True).encode()).hexdigest()

    if identity_hash != manifest.get("run_hash"):
        raise ConfigError("run manifest snapshot mismatch")

    from lexverse.job import Job

    job = Job.load(run_root.parent.parent)
    execution = next(item for item in job.executions if item.id == run_root.name)
    config = parse_config(execution.settings['config'], source_path=Path(execution.settings['source_path']), legacy=job.legacy)

    if config.hash() != manifest.get("config_hash"):
        raise ConfigError("run config snapshot mismatch")

    plugin = get_plugin(config.benchmark["name"])

    for entry in manifest.get("data_sources", []):
        path = Path(entry["path"])
        if not path.is_file() or _file_hash(path) != entry["sha256"]:
            raise ConfigError(f"shared dataset missing or changed: {path}")

    references = [{"id": task.id, "task_type": task.source.task_type, "sample_id": task.source.sample_id}
                  for task in job.tasks]

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

    if (config.hash() != manifest["config_hash"] or bundle.content_hash(legacy=config.legacy) != manifest["bundle_hash"]
            or bundle.benchmark != manifest["benchmark"] or len(bundle.tasks) != manifest["sample_count"]):
        raise ConfigError("run config/task snapshot mismatch")

    manifest["bundle"] = bundle.to_dict(legacy=config.legacy)

    return config, plugin, bundle, manifest


async def _verify_run(config, plugin, bundle, run_root, runtime, verifier, *, records):
    request_timeout = config.verification.get("model_request_timeout_sec")

    if runtime.request_timeout != request_timeout:
        runtime.stop_worker()

    runtime.request_timeout = request_timeout
    runtime.startup_timeout = config.verification.get("model_startup_timeout_sec", 1800)
    verifier_context = plugin.build_verification_context(config)

    if plugin.name == "j1bench":
        runtime.validate_phase(config.models.verifiers)
        scoring = runtime.provider(config.models.verifiers[0], label="models.verifiers.0")
        verifier_context.update(
            scoring_api_key=scoring.api_key, scoring_api_base=scoring.base_url,
            scoring_provider=scoring,
            request_timeout_sec=request_timeout,
            scoring_model=scoring.model, scoring_parameters=scoring.default_config,
            python_executable=sys.executable,
        )

    if plugin.name == "plawbench":
        runtime.validate_phase(config.models.verifiers)
        judge = runtime.provider(config.models.verifiers[0], label="models.verifiers.0")
        verifier_context.update(
            judge_provider=judge,
            judge_model=config.models.verifiers[0].name,
            judge_config={**asdict(config.models.verifiers[0]),
                          "endpoint_sha256": scoring_identity(judge)["endpoint_sha256"]},
        )

    if plugin.name in {"dlawbench", "legalworld"}:
        models = verifier_context.pop("judge_models")
        runtime.validate_phase(list(models.values()))
        verifier_context.update(
            judge_connections={key: runtime.provider(model, label=f"models.verifiers.{config.models.verifiers.index(model)}")
                               for key, model in models.items()},
            timeout_sec=config.verification.get("timeout_sec"),
        )

    if plugin.name in {"simucourt", "maser"}:
        models = plugin.verification_models(config)
        runtime.validate_phase(list(models.values()))
        connections = {key: runtime.provider(model, label=f"models.verifiers.{config.models.verifiers.index(model)}")
                       for key, model in models.items()}
        verifier_context = plugin.configure_verification_context(config, connections)

    verification_started = time.monotonic()
    logger.info("[verification] start")

    try:
        verifier_context.update(
            concurrency=int(config.verification.get("concurrency", config.raw["concurrency"])),
            verification_config={key: value for key, value in config.verification.items() if key != "concurrency"},
        )
        verified = await verifier.verify_run(
            bundle.tasks, records, run_dir=run_root,
            model_name=_tested_model(config).name, context=verifier_context, legacy=config.legacy,
        )
    except Exception as exc:
        logger.error("[verification] failed error=%s elapsed=%.1fs",
                     type(exc).__name__, time.monotonic() - verification_started)
        verification_state = _read_json(run_root / ("evaluation.json" if config.legacy else "verification.json"))
        verification_state.update(status="failed", error_type=type(exc).__name__)
        atomic_write_json(run_root / ("evaluation.json" if config.legacy else "verification.json"), verification_state)
        raise

    verification_data = verified.to_dict()
    task_metrics = verified.task_metrics()
    logger.info("[verification] %s elapsed=%.1fs scored=%s/%s",
                verification_data["status"], time.monotonic() - verification_started,
                len(verification_data["scored_task_ids"]), len(bundle.tasks))

    return task_metrics


def execution_config(config):
    return ExecutionConfig(config.agent, asdict(_tested_model(config)),
                           {'config': config.raw, 'source_path': str(config.source_path)})


def restore(job):
    configs = {item.id: parse_config(item.settings['config'], source_path=Path(item.settings['source_path']), legacy=job.legacy)
               for item in job.executions}
    plugin = get_plugin(job.tasks[0].source.name)
    bundle = TaskBundle(benchmark=plugin.name, source_version=plugin.source_version, tasks=job.tasks)

    return configs, plugin, bundle


def generation_models(config, benchmark, tasks):
    plugin = get_plugin(benchmark)
    return plugin.generation_models(config, TaskBundle(benchmark=benchmark, source_version=plugin.source_version, tasks=tasks))


def agent_spec(config, run, *, messages=None, template=None):
    from lexverse.agents.base import AgentExecutionSpec

    environment = config.raw.get('environment', {})
    environment_config = {
        target: environment[source]
        for source, target in (('override_cpus', 'cpus'), ('override_memory_mb', 'memory_mb'))
        if environment.get(source) is not None
    }

    return AgentExecutionSpec(run_dir=run, thread_id=run.name, messages=messages or [],
        model=config.models.default, selection=config.resources.model_copy(deep=True),
        retrieval=config.retrieval.model_copy(deep=True), limits=config.limits.model_copy(deep=True),
        interactive=False, mode="benchmark", environment=environment.get("type", "docker"),
        network_mode="public", allowed_hosts=[],
        environment_config=environment_config,
        context_window_tokens=config.generation.get("context_window_tokens"),
        skills_root=template / ".internal/agent_fs/skills" if template else None)


async def prepare_capabilities(config, plugin, bundle, manifest, root, *, resume):
    from lexverse.trial.worker import run_worker
    from lexverse.capabilities.base import execution_conditions

    dataset = plugin.task_inputs(config, list(dict.fromkeys(task.source.task_type for task in bundle.tasks)))
    records = {}

    for kind in dict.fromkeys(task.source.task_type for task in bundle.tasks):
        template = root / ".internal/capabilities" / kind
        spec = agent_spec(config, template)
        protected = [path.resolve().parent for path in dataset.values()]
        if hasattr(plugin, "upstream"):
            protected.extend([plugin.upstream.pristine_dir, plugin.upstream.cache_dir])
        if any(source.is_relative_to(path) or path.is_relative_to(source)
               for path in protected for source in spec.selection.knowledge):
            raise ConfigError("Knowledge must not include benchmark dataset files")
        await run_worker(spec, config.agent, prepare_only=True, task_id=kind)
        saved = json.loads((template / ".internal/manifest.json").read_text())
        records[kind] = {"resource_hash": saved["resource_hash"], "execution": execution_conditions(saved["execution"]),
                         "connection_sha256": saved["connection_sha256"]}

    conditions = {kind: {**record, "execution": execution_conditions(record["execution"])}
                  for kind, record in records.items()}
    digest = identity(conditions)

    if resume:
        previous = {kind: {**record, "execution": execution_conditions(record["execution"])}
                    for kind, record in manifest.get("capabilities", {}).items()}
        if previous != conditions or not manifest.get("capability_identity"):
            raise ConfigError("Capability resources or execution changed; start a new run")
        digest = manifest["capability_identity"]

    manifest.update(capability_identity=digest, capabilities=records)


async def execute_agent(trial, config, plugin, runtime, resource_root, work, providers):
    from lexverse.agents.static import run as run_static

    if trial.execution.agent == 'static':
        key = json.dumps(asdict(config.models.default), sort_keys=True)
        if key not in providers:
            providers[key] = runtime.provider(config.models.default, label='models.default')
        return await run_static(trial.task, providers[key])

    if trial.execution.agent == 'native':
        models = generation_models(config, plugin.name, [trial.task])
        runtime.validate_phase(list(models.values()))
        runner = plugin.create_native_runner(config)
        runner.connections = {role: runtime.provider(model, label='models.default' if role == 'default' else 'models.summary' if role == 'summary' else f'models.roles.{role}') for role, model in models.items()}
        try:
            await runner.prepare({'run_root': str(resource_root)})
            return await runner.run(trial.task, work)
        finally:
            await runner.close()

    from lexverse.trial.worker import run_worker
    from lexverse.models.trial.result import EnvironmentResult

    run = work / 'agent'
    template = resource_root / '.internal/capabilities' / trial.task.source.task_type
    messages = list(trial.task.input.messages or [{'role': 'user', 'content': trial.task.input.prompt}])
    spec = agent_spec(config, run, messages=messages, template=template)

    try:
        result = await run_worker(spec, config.agent, template=template, task_id=trial.task.id)
    finally:
        # The worker may have saved a partial trajectory before failing or being cancelled.
        source = run / 'trajectory.json'
        if source.is_file():
            from lexverse.models.trajectories import Trajectory
            from lexverse.agents.atif_converter import write_atif_trajectory
            try:
                write_atif_trajectory(Trajectory.model_validate_json(source.read_text()),
                                      trial.directory / 'trajectory.json')
            except (ValueError, OSError) as exc:
                import logging
                logging.getLogger(__name__).warning("Could not publish trajectory: %s", exc)

    if result['status'] != 'completed':
        raise ConfigError(f"Agent execution failed: {result['stop_reason']}: {result.get('error_message', '')}")

    messages.append({'role': 'assistant', 'content': result['answer']})

    trace_ref = result['trace_ref']
    # Vanilla writes a LexVerse event trace; external agents may only have a trajectory.
    # Do not declare a missing optional trace as an execution artifact.
    artifacts = ({'agent_trace': trace_ref}
                 if config.agent == 'vanilla' or Path(trace_ref).is_file() else {})
    trace = {'agent': config.agent, 'resource_hash': result['resource_hash']}
    if artifacts:
        trace['trace_ref'] = trace_ref
    return EnvironmentResult(status='completed', answer=result['answer'], messages=messages,
        dialog_history=[{'actor': trial.task.participants[0].id, 'content': result['answer']}],
        final_state={'data': {'usage': result['usage']}},
        artifacts=artifacts, trace=[trace])


async def execute_trial(trial: Trial, config, plugin, runtime, resource_root, providers=None):
    providers = {} if providers is None else providers
    # Native processes see only their private work area. Gold is retained in the trusted evaluator task.
    work = trial.directory / '.internal/native' / f'retry-{trial.retry}'
    work.mkdir(parents=True, exist_ok=True)
    task = trial.task.model_copy(deep=True)

    if trial.execution.agent != 'native':
        task.verification = None
        task.raw_payload = {}

    try:
        safe_trial = Trial(trial.id, task, trial.execution, trial.attempt, trial.directory, trial.retry,
                           trial.job_directory)
        result = await asyncio.wait_for(execute_agent(safe_trial, config, plugin, runtime, resource_root, work, providers),
                                        config.generation.get('timeout_sec'))
    except asyncio.TimeoutError as exc:
        from lexverse.trial.errors import TrialTimeoutError
        raise TrialTimeoutError('Benchmark trial timed out') from exc

    from lexverse.trial.errors import TrialError, TrialTimeoutError
    if result.status == "timeout":
        raise TrialTimeoutError("environment returned timeout")
    if result.status != "completed":
        raise TrialError(f"environment returned {result.status}")
    data = result.model_dump(mode='json')
    artifacts = {key: str((Path(path) if Path(path).is_absolute() else work / path).resolve())
                 for key, path in data['artifacts'].items()}

    for name, filename in list(artifacts.items()):
        path = Path(filename)
        if path.is_file():
            relative = path.relative_to(work) if path.is_relative_to(work) else Path(identity(name)[:12] + '-' + path.name)
            destination = trial.directory / 'artifacts' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            artifacts[name] = str(destination)

    trajectory = {'messages': data['messages'], 'dialog_history': data['dialog_history'],
                  'final_state': data['final_state'], 'trace': data['trace']}

    if trial.execution.agent in ('static', 'native'):
        atomic_write_json(trial.directory / 'trajectory.json', trajectory)

    return {'id': trial.task.id, 'task': trial.task.source.task_type or plugin.name,
            'sample_id': trial.task.source.sample_id, 'source': trial.task.source.model_dump(mode='json'),
            'benchmark': plugin.name, 'status': result.status, 'answer': result.answer,
            'trajectory_ref': str(trial.directory / 'trajectory.json'),
            'artifacts': artifacts,
            'execution_artifact_hashes': {name: _file_hash(Path(path)) for name, path in artifacts.items()},
            'usage': data['final_state'].get('data', {}).get('usage') or {'status': 'unavailable'}}


async def verify_job(job, configs, plugin, bundle, runtimes):
    verification_data = load_verification(job)
    verification_records = verification_data['configurations']

    for execution in job.executions:
        config = configs[execution.id]
        verifier = plugin.create_verifier({**config.benchmark, 'offline': bool(config.generation.get('offline', False))})
        for attempt in range(1, job.n_attempts + 1):
            key = f'{execution.id}__attempt_{attempt}'
            directory = job.verification_directory / '.internal' / key
            directory.mkdir(parents=True, exist_ok=True)
            selected = [trial for trial in job.plan() if trial.execution.id == execution.id and trial.attempt == attempt]
            records = {trial.task.id: json.loads((trial.directory / 'result.json').read_text()) for trial in selected}
            for record in records.values():
                for field in ('artifacts', 'execution_artifacts'):
                    record[field] = {name: os.path.relpath(path, directory) for name, path in record.get(field, {}).items()}
            for record in records.values():
                record['response'] = record.get('answer')
                record['execution_artifacts'] = record.get('artifacts', {})
            try:
                await _verify_run(config, plugin, bundle, directory, runtimes[execution.id], verifier,
                                    records=records)
            finally:
                saved = directory / ('evaluation.json' if job.legacy else 'verification.json')
                ledger = json.loads(saved.read_text()) if saved.exists() else {'status': 'failed'}
                verification_records[key] = {name: value for name, value in ledger.items() if name not in {'units', 'trials'}}
                for group in verification_records[key].get('groups', []):
                    group.pop('raw_result', None)
                verification_records[key]['native_ref'] = str(saved.relative_to(job.directory))
                verification_records[key]['work_dir'] = str(directory.relative_to(job.directory))
                for trial in selected:
                    record = records[trial.task.id]
                    ledger = verification_records[key]
                    if trial.task.id in ledger.get('scored_task_ids', []) and not record.get('verifier_result'):
                        record['verifier_result'] = {
                            'status': 'completed', 'granularity': 'native_group',
                            'group_refs': [job.build_verification_ref(f'/configurations/{key}/groups/{index}')
                                           for index, group in enumerate(ledger.get('groups', []))
                                           if trial.task.id in group.get('task_ids', [])],
                        }
                    for field in ('artifacts', 'execution_artifacts'):
                        record[field] = {name: str((directory / path).resolve()) for name, path in record.get(field, {}).items()}
                    verification_record = record.get('verifier_result') or {}
                    if verification_record.get('artifacts'):
                        verification_record['artifacts'] = {name: str((directory / path).resolve())
                                                   for name, path in verification_record['artifacts'].items()}
                    verification_record.pop('raw_result', None)
                    verification_data['trials'][trial.id] = verification_record or {'status': 'pending'}
                aggregator = plugin.create_aggregator()
                aggregate = aggregator.aggregate(list(records.values()), model_name=_tested_model(config).name)
                for group in verification_records[key].get('groups', []):
                    if group['task_type'] in aggregate['tasks']:
                        aggregate['tasks'][group['task_type']]['metrics'] = group['metrics']
                extra = plugin.summary_metadata(aggregate)
                if set(extra) & (set(verification_records[key]) | {"aggregate"}):
                    raise ValueError("plugin summary metadata cannot overwrite verification fields")
                verification_records[key].update(extra)
                verification_records[key]['aggregate'] = aggregate
                (directory / 'evaluation_result.csv').unlink(missing_ok=True)
                save_verification(job, verification_data)
                write_summary(job)
                write_verification_report(job)

    return verification_records


async def _execute_job(job, configs, plugin, bundle, *, resume, verify, excel=False, manifests=None, trial_id=None):
    from lexverse.benchmarks.sources import prepare_run_sources

    # Validate scoring configuration before preparing or executing Trials.
    if verify:
        for config in configs.values():
            plugin.build_verification_context(config)
            if plugin.name in {"simucourt", "maser"}:
                plugin.verification_models(config)

    # Keep model runtimes, resource roots, and providers per execution configuration.
    runtimes, roots, providers = {}, {}, {}
    executions, generation_tasks = job.executions, bundle.tasks
    if trial_id is not None and not verify:
        selected = [trial for trial in job.plan() if trial.id == trial_id]
        execution_ids = {trial.execution.id for trial in selected}
        executions = [item for item in job.executions if item.id in execution_ids]
        generation_tasks = [trial.task for trial in selected]

    # Prepare resources and validate saved snapshots before scheduling Trials.
    async def prepare():
        logging.getLogger('lexverse.run').info('[run] preparing sources')

        for execution in executions:
            config = configs[execution.id]
            root = job.directory / '.internal' / execution.id
            root.mkdir(parents=True, exist_ok=True)
            roots[execution.id] = root
            providers[execution.id] = {}
            runtime = ModelRuntime(root, startup_timeout=config.generation.get('model_startup_timeout_sec', 1800),
                                   request_timeout=config.generation.get('model_request_timeout_sec'))
            runtime.validate_phase(list(generation_models(config, plugin.name, generation_tasks).values()))
            runtimes[execution.id] = runtime
            manifest_path = root / 'manifest.json'
            if resume:
                restored_config, _, restored_bundle, manifest = load_run_snapshot(root)
                if restored_config.hash() != config.hash() or restored_bundle.content_hash(legacy=job.legacy) != bundle.content_hash(legacy=job.legacy):
                    raise ConfigError('Job tasks/configuration differ from source snapshot')
            else:
                manifest = manifests[execution.id]
            if config.agent not in {'static', 'native'}:
                await prepare_capabilities(config, plugin, bundle, manifest, root, resume=resume)
            manifest["run_hash"] = prepare_run_sources(root, manifest, plugin, resume=resume,
                                offline=bool(config.generation.get('offline', False)), config=config)
            atomic_write_json(manifest_path, {key: value for key, value in manifest.items() if key != 'bundle'})
            runtime.bind_manifest(manifest)

    # Connect the shared Job scheduler to benchmark-specific Trial execution.
    async def execute(trial):
        return await execute_trial(trial, configs[trial.execution.id], plugin,
                                   runtimes[trial.execution.id], roots[trial.execution.id], providers[trial.execution.id])

    report_ready = False
    try:

        # Execute Trials through the shared Job lifecycle.
        summary = await run_job(job, execute, resume=resume, prepare=prepare, trial_id=trial_id)

        # Score saved answers and refresh the Job summary.
        if verify:
            await verify_job(job, configs, plugin, bundle, runtimes)
            report_ready = True
            summary = summarize(job)

        # Export the optional per-Trial workbook.
        if excel:
            export_excel(job)
        return summary
    finally:

        # Keep per-group reports; rebuild on interruption before scoring finishes.
        for runtime in runtimes.values():
            await runtime.close()
        if verify and not report_ready and (job.directory / 'job.json').is_file():
            write_verification_report(job)


def command(config, *, verify, dry_run=False):
    if config._resolved_resources is None:
        config.resolve(Path.cwd())

    # Validate overrides before building the benchmark execution plan.
    if config.verifier.override_timeout_sec is not None:
        raise ConfigError('Benchmark scoring timeouts belong in plugin verification settings')

    # Resolve each Agent configuration and require the same task selection.
    configs, manifests, executions = {}, {}, []
    for agent_index, agent in enumerate(config.agents):
        if agent.override_timeout_sec is not None and agent.name in {'static', 'native'}:
            raise ConfigError(f'{agent.name} does not support Agent timeout overrides')
        if (agent.name in {'static', 'native'} and (config.environment.override_cpus is not None
                or config.environment.override_memory_mb is not None)):
            raise ConfigError(f'{agent.name} does not use a Task sandbox for CPU or memory overrides')
        resolved = resolve_benchmark_config(config, agent_index, source_path=Path.cwd() / "job.yaml")
        current_plugin, current_bundle, manifest = build_run(resolved)
        if executions and current_bundle.model_dump(mode="json") != bundle.model_dump(mode="json"):
            raise ConfigError("Agent configurations must select identical benchmark tasks")
        plugin, bundle = current_plugin, current_bundle
        execution = execution_config(resolved)
        executions.append(execution)
        configs[execution.id], manifests[execution.id] = resolved, manifest

    # Choose a fresh result directory and construct the shared Job.
    root = config.output_dir or Path("runs/benchmark_tasks") / plugin.name / new_run_id()
    if root.exists() and any(root.iterdir()):
        raise ConfigError("Job directory is not empty; choose a new directory")
    job = Job(bundle.tasks, executions, root.resolve(), n_attempts=config.attempts,
              concurrency=config.concurrency, max_retries=config.max_retries)
    print(f'Planned Trials: {len(job.plan())}', file=sys.stderr)

    # Print the plan and stop when execution is not requested.
    if dry_run:
        print(json.dumps({"planned_trials": len(job.plan()), "job_dir": str(job.directory)}))
        return 0

    # Run generation and optional scoring with scoped terminal and file logging.
    with _run_logging(job.directory):
        logging.getLogger('lexverse.run').info('[run] preparing benchmark=%s resume=%s', plugin.name, False)
        summary = asyncio.run(_execute_job(job, configs, plugin, bundle, resume=False, verify=verify,
                                  excel=config.export_excel, manifests=manifests))
        logging.getLogger('lexverse.run').info('[run] %s success=%s failed=%s directory=%s', summary['status'],
            summary['statuses'].get('completed', 0), summary['statuses'].get('failed', 0), job.directory)

    # Return a process exit code based on execution and scoring completion.
    scored = summary['verification_status'] in {'completed', 'not_configured'}
    return 0 if summary['execution_status'] == 'completed' and (not verify or scored) else 4


def resume_saved(directory: Path, *, verify=True):
    return _resume_job(Job.load(directory.resolve()), verify=verify)


def _resume_job(job, *, verify=True):
    configs, plugin, bundle = restore(job)

    with _run_logging(job.directory):
        summary = asyncio.run(_execute_job(job, configs, plugin, bundle, resume=True, verify=verify))

    scored = summary["verification_status"] in {"completed", "not_configured"}

    return 0 if summary["execution_status"] == "completed" and (not verify or scored) else 4


def verify_saved(directory):
    return _verify_saved_job(Job.load(directory.resolve()))


def _verify_saved_job(job):
    configs, plugin, bundle = restore(job)
    runtimes = {item.id: ModelRuntime(job.directory / '.internal' / item.id) for item in job.executions}

    async def execute():
        try:
            for item in job.executions:
                root = job.directory / '.internal' / item.id
                manifest = json.loads((root / 'manifest.json').read_text())
                from lexverse.benchmarks.sources import prepare_run_sources
                prepare_run_sources(root, manifest, plugin, resume=True, offline=True, config=configs[item.id])
                runtimes[item.id].bind_manifest(manifest)
            return await verify_job(job, configs, plugin, bundle, runtimes)
        finally:
            for runtime in runtimes.values():
                await runtime.close()
    verification_records = asyncio.run(execute())

    return 0 if all(item['status'] == 'completed' for item in verification_records.values()) else 4
