import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from uuid import uuid4

from lexverse.config import ConfigError, ModelConfig, SecretsMissing, load_profile
from lexverse.capabilities.base import execution_conditions
from lexverse.runtime.results import atomic_write_json
from lexverse.runtime.dependencies import requires_agent_dependencies
from lexverse.runtime.paths import CAPABILITIES_ROOT


def enabled(config):
    return config is not None and config.generation.get("capabilities", {}).get("enabled", False)


async def run_worker(payload, control_dir):
    from lexverse.capabilities.benchmark import BenchmarkCapabilities

    settings = BenchmarkCapabilities.model_validate(payload["capabilities"])
    selection, _ = settings.effective(payload["task_type"])
    control_dir.mkdir(parents=True, exist_ok=True)
    request = control_dir / "request.json"
    error = control_dir / "error.json"
    payload = {**payload, "reply_path": str(error),
               "dependencies_required": requires_agent_dependencies(settings.agent, selection.knowledge)}
    error.unlink(missing_ok=True)
    atomic_write_json(request, payload)
    with (control_dir / "worker.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "lexverse.runtime.worker", "benchmark", str(request),
            stdout=log, stderr=log,
        )
        try:
            await process.wait()
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
    if process.returncode:
        message = json.loads(error.read_text())["error"] if error.exists() else f"Agent worker failed; see {control_dir / 'worker.log'}"
        raise ConfigError(message)
    return json.loads((Path(payload["run_dir"]) / "result.json").read_text())


async def prepare_capabilities(config, plugin, bundle, manifest, root, *, resume):
    settings = config.generation["capabilities"]
    if settings.get("task_overrides", {}).keys() - set(plugin.task_types()):
        raise ConfigError("Unknown capability task override")
    from lexverse.capabilities.benchmark import BenchmarkCapabilities
    parsed = BenchmarkCapabilities.model_validate(settings)
    dataset = plugin.task_inputs(config, list(dict.fromkeys(task.source.task_type for task in bundle.tasks)))
    records = {}
    for kind in dict.fromkeys(task.source.task_type for task in bundle.tasks):
        selection, _ = parsed.effective(kind)
        protected = [path.resolve().parent for path in dataset.values()]
        if hasattr(plugin, "upstream"):
            protected.extend([plugin.upstream.pristine_dir, plugin.upstream.cache_dir])
        if any(source.is_relative_to(path) or path.is_relative_to(source)
               for path in protected for source in selection.knowledge):
            raise ConfigError("Knowledge must not include benchmark dataset files")
        template = root / ".internal/capabilities" / kind
        if resume and not template.exists():
            template = root / "capabilities" / kind
        await run_worker({
            "operation": "prepare", "run_dir": str(template), "task_id": kind, "task_type": kind,
            "capabilities": settings, "model": asdict(config.models.default),
        }, template / ".internal/control")
        saved = json.loads((template / ".internal/manifest.json").read_text())
        records[kind] = {"resource_hash": saved["resource_hash"], "execution": saved["execution"],
                         "connection_sha256": saved["connection_sha256"]}
    conditions = {kind: {**record, "execution": execution_conditions(record["execution"])}
                  for kind, record in records.items()}
    identity = hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()
    if resume:
        previous = {kind: {**record, "execution": execution_conditions(record["execution"])}
                    for kind, record in manifest.get("capabilities", {}).items()}
        if previous != conditions or not manifest.get("capability_identity"):
            raise ConfigError("Capability resources or execution changed; start a new run")
        identity = manifest["capability_identity"]
    manifest["capability_identity"] = identity
    manifest["capabilities"] = records


async def act(config, observation, context):
    run = Path(context["work_dir"]) / "agent" / uuid4().hex
    template = Path(context["run_root"]) / ".internal/capabilities" / context["task_type"]
    if not template.exists():
        template = Path(context["run_root"]) / "capabilities" / context["task_type"]
    result = await run_worker({
        "operation": "execute", "run_dir": str(run), "task_id": context["task_id"],
        "task_type": context["task_type"], "messages": observation,
        "template": str(template),
        "capabilities": config.generation["capabilities"], "model": asdict(config.models.default),
    }, run / ".internal/control")
    return result


# Worker-side enhancement execution; Agent imports stay inside these functions.
def initialize(run, payload):
    internal = run / ".internal"
    internal.mkdir(parents=True, exist_ok=True)
    for name in (("inputs", "skills", "workspace") if payload["operation"] == "prepare" else ("inputs", "workspace")):
        (internal / "agent_fs" / name).mkdir(parents=True)
    (run / "artifacts").mkdir()
    atomic_write_json(internal / "manifest.json", {
        "run_id": run.name, "task_id": payload["task_id"], "participant_id": "assistant",
        "attempt_id": run.name, "thread_id": run.name,
    })
    atomic_write_json(run / "result.json", {"status": "running", "usage": {}})


def check_sources(run, selection, cache):
    from lexverse.capabilities.knowledge.loaders import scan_source
    from lexverse.capabilities.knowledge.index import is_index_directory, verify_index
    from lexverse.capabilities.registry import ResourceRegistry, load_registry
    from lexverse.capabilities.skills import prepare_skills

    saved = json.loads((run / ".internal/manifest.json").read_text())["resources"]
    registry = load_registry(selection.registry)
    if registry.model_dump(mode="json") != ResourceRegistry.model_validate(saved["registry"]).model_dump(mode="json"):
        raise ConfigError("Capability registry changed; start a new run")
    for source, record in zip(selection.knowledge, saved["knowledge"], strict=True):
        manifest = json.loads((Path(record["path"]) / "manifest.json").read_text())
        if is_index_directory(source):
            unchanged = verify_index(source)["index_hash"] == manifest["index_hash"]
        else:
            unchanged = scan_source(source)[0] == manifest["identity"]["files"]
        if not unchanged:
            raise ConfigError("Knowledge source changed; start a new run")
    with tempfile.TemporaryDirectory() as temporary:
        current = prepare_skills(registry, selection.skills, cache, Path(temporary))
        if current["files"] != saved["skills"]["files"]:
            raise ConfigError("Skill source changed; start a new run")


async def execute_benchmark(payload):
    """Prepare or execute an enhancement request after worker dependency activation."""
    from lexverse.agents.general_runner import execute_general_agent
    from lexverse.agents.spec import AgentExecutionSpec
    from lexverse.capabilities.preparation import prepare_agent_resources
    from lexverse.capabilities.benchmark import BenchmarkCapabilities
    from lexverse.runtime.identity import execution_identity
    from lexverse.tasks.user import OutputContract

    settings = BenchmarkCapabilities.model_validate(payload["capabilities"])
    selection, limits = settings.effective(payload["task_type"])
    run = Path(payload["run_dir"])
    manifest_path = run / ".internal/manifest.json"
    cache = CAPABILITIES_ROOT
    preparing = payload["operation"] == "prepare"
    profile = load_profile(payload["model"]["profile"])
    connection_hash = hashlib.sha256((profile.base_url or "https://api.openai.com/v1").encode()).hexdigest()
    model = ModelConfig(**payload["model"])
    if preparing and settings.agent == "lexverse":
        from lexverse.providers.api import create_chat_model
        create_chat_model(model, context_window_tokens=settings.context_window_tokens)
    if preparing and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["connection_sha256"] != connection_hash:
            raise ConfigError("Model endpoint changed; start a new run")
        check_sources(run, selection, cache)
    else:
        initialize(run, payload)
        manifest = json.loads(manifest_path.read_text())
        if not preparing:
            template = Path(payload["template"])
            frozen = json.loads((template / ".internal/manifest.json").read_text())
            if frozen["connection_sha256"] != connection_hash:
                raise ConfigError("Model endpoint changed; start a new run")
            manifest["resources"] = frozen["resources"]
            manifest["resource_hash"] = frozen["resource_hash"]
    manifest["connection_sha256"] = connection_hash
    atomic_write_json(manifest_path, manifest)
    spec = AgentExecutionSpec(
        run_dir=run, thread_id=run.name, messages=payload.get("messages", []), model=model,
        selection=selection, retrieval=settings.retrieval, limits=limits,
        outputs=OutputContract(), interactive=False, mode="benchmark",
        environment=settings.environment,
        network_mode=settings.network_mode, allowed_hosts=settings.allowed_hosts,
        context_window_tokens=settings.context_window_tokens,
        skills_root=None if preparing else Path(payload["template"]) / ".internal/agent_fs/skills",
    )
    default_env_dir = Path(__file__).resolve().parents[2] / "user_workspace/task/example/environment"
    if preparing:
        # Build the shared template once, without starting an agent.
        await prepare_agent_resources(spec, agent_name=settings.agent, cache=cache)
        result = {"status": "completed", "stop_reason": "resources_prepared", "run_dir": str(run),
                  "answer": None, "artifacts": [], "question": None,
                  "usage": json.loads((run / "result.json").read_text()).get("usage", {}),
                  "trace_ref": str(run / ".internal/events.jsonl")}
    else:
        # Each sample uses the template records without preparing resources again.
        execution = execution_identity(spec, agent_name=settings.agent)
        expected = execution_conditions(frozen["execution"])
        current = execution_conditions(execution)
        # Sample messages differ; model, resources and execution limits stay pinned.
        expected.pop("messages_hash", None)
        current.pop("messages_hash", None)
        if expected != current:
            raise ConfigError("Execution configuration changed since resource preparation; start a new run")
        manifest["execution"] = execution
        atomic_write_json(manifest_path, manifest)
        result = await execute_general_agent(
            spec, resources=manifest["resources"], agent_name=settings.agent, cache=cache,
            environment_dir=default_env_dir if default_env_dir.is_dir() else None,
        )
    if result["status"] != "completed":
        raise ConfigError(f"Capability execution failed: {result['stop_reason']}: {result.get('error_message', result.get('error_type', ''))}")
    result["resource_hash"] = json.loads(manifest_path.read_text())["resource_hash"]
    atomic_write_json(run / "result.json", result)
    return result


def run_benchmark(payload):
    """Synchronous benchmark entry called by the shared worker."""
    try:
        asyncio.run(execute_benchmark(payload))
        return 0
    except (ConfigError, SecretsMissing, ValueError, OSError) as exc:
        atomic_write_json(Path(payload["reply_path"]), {"error": str(exc)})
        return 2
