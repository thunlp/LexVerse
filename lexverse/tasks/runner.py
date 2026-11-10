from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import shutil

from lexverse.models.trial.paths import RunPaths, new_run_id
from lexverse.agents.base import AgentExecutionSpec
from lexverse.config import ConfigError
from lexverse.utils.files import atomic_write_json


def prepare_user_run(task, trial_dir: Path, *, task_source: Path) -> Path:
    run = trial_dir.resolve()
    internal = run / ".internal"

    for name in ("workspace", "skills"):
        (internal / "agent_fs" / name).mkdir(parents=True, exist_ok=True)

    (run / "artifacts").mkdir(exist_ok=True)
    atomic_write_json(internal / "manifest.json", {
        "schema_version": "1.0", "run_id": run.name, "task_id": task.id,
        "thread_id": new_run_id(), "task_source": str(task_source),
    })
    atomic_write_json(RunPaths(run).worker_result, {
        "status": "running", "stop_reason": None, "run_dir": str(run),
        "answer": None, "artifacts": [], "question": None, "usage": {},
        "trace_ref": str(internal / "events.jsonl"),
    })

    return run


def execution_spec(trial) -> AgentExecutionSpec:
    run = trial.directory
    paths = RunPaths(run)
    manifest = json.loads((paths.internal / "manifest.json").read_text())
    from lexverse.capabilities.resources import ResourceSelection
    from lexverse.models.task.config import RetrievalConfig, AgentLimits
    from lexverse.config import ModelConfig

    task = trial.task.model_copy(deep=True)
    execution = trial.execution
    from lexverse.capabilities.resources import MCPService, merge_mcp

    resources = execution.settings.get("resources", {})
    servers = merge_mcp([MCPService.model_validate(item) for item in task.capabilities.get("mcp_servers", [])],
                        [MCPService.model_validate(item) for item in resources.get("mcp_servers", [])])

    return AgentExecutionSpec(run_dir=run, thread_id=manifest["thread_id"],
        messages=[{"role": "user", "content": task.input.prompt}], model=ModelConfig(**execution.model),
        selection=ResourceSelection(knowledge=list(dict.fromkeys([*task.capabilities.get("knowledge", []),
                                                                  *resources.get("knowledge", [])])),
                                   skills=resources.get("skills", []), mcp_servers=servers,
                                   disabled_tools=resources.get("disabled_tools", []),
                                   **({"tools": resources["tools"]} if "tools" in resources else {})),
        retrieval=RetrievalConfig.model_validate(execution.settings.get("retrieval", {})),
        limits=AgentLimits.model_validate(execution.settings.get("limits", {})),
        interactive=execution.settings["interactive"], environment=execution.settings["environment"],
        network_mode=task.environment["network_mode"], allowed_hosts=task.environment["allowed_hosts"],
        context_window_tokens=execution.settings.get("context_window_tokens"),
        environment_config=task.environment, artifacts=task.outputs.get("artifacts", []),
        task_skills=task.capabilities.get("skills", []),
        verifier=task.verification.config if task.verification is not None and execution.settings.get('verify', True) else None)



def main(payload, *, job_trial=None):
    args = argparse.Namespace(**payload["args"])

    try:
        from lexverse.job import load_trial
        job, trial = job_trial if job_trial is not None else load_trial(Path(args.run_dir))
        # Load the saved result before resuming the Agent in a subprocess.
        prior = json.loads((Path(args.run_dir) / "result.json").read_text()) if args.command == "resume" else {}

        # Pass the resume request to the worker and refresh Job records afterward.
        with tempfile.TemporaryDirectory(prefix="lexverse-resume-") as temporary:
            request = Path(temporary) / "request.json"
            error = Path(temporary) / "error.json"
            atomic_write_json(request, {"resume": {"reply": args.reply, "confirm_facts": args.confirm_facts},
                                        "spec": execution_spec(trial).to_dict(),
                                        "reply_path": str(error), "dependencies_required": True})
            try:
                process = subprocess.run([sys.executable, "-m", "lexverse.trial.worker", str(request)])
                if process.returncode:
                    raise ConfigError(json.loads(error.read_text())["error"] if error.exists() else
                                      "Agent resume worker did not complete")
            finally:
                result = _refresh_job(job, trial, prior)

        # Print the refreshed result and report the resume outcome.
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"completed", "needs_input"} and not result.get("resume_error") else 1
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Task preparation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _refresh_job(job, trial, prior):
    from lexverse.job import write_verification_report

    trial_dir = trial.directory
    record = json.loads(RunPaths(trial_dir).worker_result.read_text())

    if record.get('resume_error'):
        return record

    required = ["artifacts/" + item["destination"] for item in trial.task.outputs.get("artifacts", [])
                if item.get("required", True)]
    annotate_result(trial_dir, record, required, trial.task.verification is not None)
    record.update(trial.identity_record(job.id), retry=prior.get('retry', 0))
    job.save_trial_result(trial, record)
    write_verification_report(job)

    return record


def command(config, *, verify=True, dry_run=False):
    from lexverse.tasks.user import load_user_task

    # Load the task package and execute its Job through the async runner.
    task = load_user_task(config.path)
    result = asyncio.run(_execute_job(config, task, verify=verify, dry_run=dry_run))

    # Print the Job result and return the process exit code.
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0 if dry_run or result["status"] in {"completed", "needs_input"} or (
        not verify and result['execution_status'] == 'completed') else 1


async def _execute_job(config, task, *, verify=True, dry_run=False):
    from lexverse.job import Job, run_job, export_excel, write_verification_report
    from lexverse.models.trial.config import ExecutionConfig
    from lexverse.tasks.builders import user_task

    # Build the public task and validate merged MCP resources.
    source = config.path if config.path.is_dir() else config.path.parent
    public = user_task(task, source=source)
    from lexverse.capabilities.resources import MCPService, merge_mcp

    merge_mcp([MCPService.model_validate(item) for item in public.capabilities.get('mcp_servers', [])],
              [MCPService.model_validate(item) for item in config.resources.get('mcp_servers', [])])

    # Apply environment and verifier overrides without changing Task scoring rules.
    judge_models = config.models.get('verifiers', [])
    if judge_models:
        if len(judge_models) != 1:
            raise ConfigError('User Tasks accept one model in models.verifiers')
        if public.verification is None or public.verification.config['environment_mode'] != 'separate':
            raise ConfigError('User Task Judge models require separate verification')
        public.verification.config['model'] = dict(judge_models[0])
    if config.verifier.override_timeout_sec is not None and public.verification is None:
        raise ConfigError('Verifier timeout override requires enabled Task verification')

    if config.environment.override_cpus is not None:
        public.environment['cpus'] = config.environment.override_cpus

    if config.environment.override_memory_mb is not None:
        public.environment['memory_mb'] = config.environment.override_memory_mb

    if public.verification and config.verifier.override_timeout_sec is not None:
        public.verification.config['timeout_sec'] = config.verifier.override_timeout_sec

    # Build one execution configuration per selected Agent.
    executions = [ExecutionConfig(agent.name, agent.model or config.models["default"], {
        **config.generation, "interactive": config.interactive, "environment": config.environment.type,
        "verify": verify,
        "resources": config.resources,
        "limits": {**config.generation.get('limits', {}),
                   **({'active_timeout_sec': agent.override_timeout_sec or task.agent.timeout_sec}
                      if agent.override_timeout_sec or task.agent.timeout_sec else {})},
    }) for agent in config.agents]

    # Construct the Job and validate its output directory and execution plan.
    root = config.output_dir or Path("runs/user_tasks") / task.id / new_run_id()
    job = Job([public], executions, root.resolve(), n_attempts=config.attempts,
              concurrency=config.concurrency, max_retries=config.max_retries)

    if job.directory.is_relative_to(source):
        raise ConfigError("Job directory must not be inside the task package")

    if config.interactive and len(job.plan()) > 1:
        raise ConfigError("Multiple interactive Trials are unsupported; resume Trials individually")

    print(f'Planned Trials: {len(job.plan())}', file=sys.stderr)
    if dry_run:
        return {"planned_trials": len(job.plan()), "job_dir": str(job.directory)}

    if job.directory.exists() and any(job.directory.iterdir()):
        raise ConfigError("Job directory is not empty; choose a new directory")

    # Connect the shared Job scheduler to User Task execution.
    async def execute(trial):
        return await execute_trial(job, trial, verify=verify)

    # Snapshot the task package before executing Trials.
    async def prepare():
        asset_root = job.directory / ".internal/task"
        asset_root.mkdir(parents=True)

        for name in ("task.toml", "instruction.md", "environment", "tests", "solution"):
            path = source / name
            if path.is_dir():
                shutil.copytree(path, asset_root / name)
            elif path.is_file():
                shutil.copy2(path, asset_root / name)

    # Run the Job and refresh reports even if execution raises an error.
    try:
        result = await run_job(job, execute, prepare=prepare)
    finally:
        if (job.directory / 'job.json').is_file():
            write_verification_report(job)

    # Export the optional workbook and return the Job result.
    if config.export_excel:
        export_excel(job)

    return {**result, "run_dir": str(job.directory)}


async def execute_trial(job, trial, *, verify=True):
    # Archive previous execution files when retrying this Trial.
    if (trial.directory / ".internal/agent_fs").exists():
        archive = trial.directory / ".internal/retries" / str(trial.retry)
        archive.mkdir(parents=True, exist_ok=False)
        for path in list((trial.directory / ".internal").iterdir()):
            if path.name not in {"lifecycle.jsonl", "retries", "result_history"}:
                shutil.move(str(path), archive / path.name)
        for name in ("artifacts", "trajectory.json"):
            path = trial.directory / name
            if path.exists():
                shutil.move(str(path), archive / name)
        (trial.directory / "artifacts").mkdir()

    # Prepare the Trial and run its selected Agent through the shared worker.
    from lexverse.trial.worker import run_worker

    prepare_user_run(trial.task, trial.directory,
                     task_source=job.directory / ".internal/task/task.toml")
    spec = execution_spec(trial)
    spec.verifier = trial.task.verification.config if verify and trial.task.verification else None
    result = await run_worker(spec, trial.execution.agent, task_id=trial.task.id)

    # Record output validation, verification results, and retry eligibility.
    annotate_result(trial.directory, result, ["artifacts/" + item["destination"] for item in spec.artifacts
                     if item.get("required", True)], trial.task.verification is not None)
    result["_retryable"] = result.get("stop_reason") == "execution_failed"

    return result


def resume_saved(directory: Path, *, verify=True) -> int:
    from lexverse.job import Job

    return _resume_job(Job.load(directory.expanduser().resolve()), verify=verify)


def _resume_job(job, *, verify=True) -> int:
    from lexverse.job import run_job, write_verification_report

    if any(task.source.kind != 'user' for task in job.tasks):
        raise ConfigError('Expected a User Task Job')

    summary = asyncio.run(run_job(job, lambda trial: execute_trial(job, trial, verify=verify), resume=True))
    if verify and summary['execution_status'] == 'completed':
        return _verify_job(job)
    write_verification_report(job)
    return 0 if summary['execution_status'] == 'completed' else 4


def verify_saved(directory: Path) -> int:
    from lexverse.job import Job

    return _verify_job(Job.load(directory.expanduser().resolve()))


def _verify_job(job) -> int:
    from lexverse.job import load_verification, save_verification, write_summary, write_verification_report
    from lexverse.trial.verification import verify_trial

    data = load_verification(job)

    async def score():
        pending = []
        for trial in job.plan():
            if trial.task.source.kind != 'user':
                raise ConfigError('Expected a User Task Job')
            if trial.task.verification is None:
                continue
            path = trial.directory / 'result.json'
            record = json.loads(path.read_text()) if path.is_file() else {'status': 'pending'}
            if record['status'] != 'completed':
                continue
            if not trial._completed_artifacts_valid(record):
                raise ConfigError('Saved Task artifacts changed; resume the Trial before evaluating')
            if data['trials'].get(trial.id, {}).get('status') == 'completed':
                continue
            if trial.task.verification.config['environment_mode'] != 'separate':
                raise ConfigError('Missing shared-environment verification requires an explicit Trial retry')
            records = json.loads((trial.directory / '.internal/artifacts.json').read_text())['artifacts']
            pending.append((trial, records))
        if pending:
            write_verification_report(job)
        for trial, records in pending:
            spec = execution_spec(trial)
            spec.verifier = trial.task.verification.config
            result = await verify_trial(spec, None, job.directory / '.internal/task/tests', records)
            data['trials'][trial.id] = _build_verification_record(result)
            save_verification(job, data)

    try:
        asyncio.run(score())
    finally:
        summary = write_summary(job)
        write_verification_report(job)
    return 0 if summary['verification_status'] in {'completed', 'not_configured'} else 4


def _build_verification_record(result):
    return {
        'status': 'completed' if result['return_code'] == 0 and result.get('rewards') is not None else 'failed',
        'metrics': result.get('rewards') or {}, 'return_code': result['return_code'],
        **{key: result[key] for key in ('error_type', 'error_message', 'phase') if key in result},
    }


def annotate_result(directory, result, required, verification_configured):
    valid = {"artifacts/" + item["name"] for item in result.get("artifacts", [])}
    missing = [name for name in required if not (directory / name).exists()
               or (not (directory / name).is_dir() and name not in valid)]
    result["outputs"] = {"status": "missing" if missing else "satisfied", "missing": missing}
    verifier = result.pop("verifier", None)

    if verifier is not None:
        result["verifier_result"] = _build_verification_record(verifier)
    else:
        result["verifier_result"] = {"status": "pending" if verification_configured else "not_configured"}

