import asyncio
import json
import hashlib
from pathlib import Path
import tempfile

from lexverse.agents.agent_loop import execute_agent
from lexverse.capabilities.base import AgentExecutionSpec
from lexverse.capabilities.benchmark import BenchmarkCapabilities
from lexverse.capabilities.knowledge.loaders import scan_source
from lexverse.capabilities.registry import load_registry
from lexverse.capabilities.skills import prepare_skills
from lexverse.config import ConfigError, ModelConfig, SecretsMissing, load_profile
from lexverse.runtime.results import atomic_write_json
from lexverse.tasks.user import OutputContract


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
    saved = json.loads((run / ".internal/manifest.json").read_text())["resources"]
    registry = load_registry(selection.registry)
    if registry.model_dump(mode="json") != saved["registry"]:
        raise ConfigError("Capability registry changed; start a new run")
    for source, record in zip(selection.knowledge, saved["knowledge"], strict=True):
        manifest = json.loads((Path(record["path"]) / "manifest.json").read_text())
        if scan_source(source)[0] != manifest["identity"]["files"]:
            raise ConfigError("Knowledge source changed; start a new run")
    with tempfile.TemporaryDirectory() as temporary:
        current = prepare_skills(registry, selection.skills, cache, Path(temporary), interactive=False)
        expected = {name: digest for name, digest in saved["skills"]["files"].items() if name != "bindings.json"}
        if current["files"] != expected:
            raise ConfigError("Skill source changed; start a new run")


async def execute(payload):
    settings = BenchmarkCapabilities.model_validate(payload["capabilities"])
    selection, limits = settings.effective(payload["task_type"])
    run = Path(payload["run_dir"])
    cache = Path(__file__).resolve().parents[2] / ".lexverse/capabilities"
    preparing = payload["operation"] == "prepare"
    profile = load_profile(payload["model"]["profile"])
    connection_hash = hashlib.sha256((profile.base_url or "https://api.openai.com/v1").encode()).hexdigest()
    if preparing:
        from lexverse.providers.chat_model import create_chat_model
        create_chat_model(ModelConfig(**payload["model"]), context_window_tokens=settings.context_window_tokens)
    if preparing and (run / ".internal/manifest.json").exists():
        manifest = json.loads((run / ".internal/manifest.json").read_text())
        if manifest["connection_sha256"] != connection_hash:
            raise ConfigError("Model endpoint changed; start a new run")
        check_sources(run, selection, cache)
    else:
        initialize(run, payload)
        if not preparing:
            template = Path(payload["template"])
            frozen = json.loads((template / ".internal/manifest.json").read_text())
            if frozen["connection_sha256"] != connection_hash:
                raise ConfigError("Model endpoint changed; start a new run")
            manifest = json.loads((run / ".internal/manifest.json").read_text())
            manifest["resources"] = frozen["resources"]
            manifest["resource_hash"] = frozen["resource_hash"]
            atomic_write_json(run / ".internal/manifest.json", manifest)
    manifest = json.loads((run / ".internal/manifest.json").read_text())
    manifest["connection_sha256"] = connection_hash
    atomic_write_json(run / ".internal/manifest.json", manifest)
    model = ModelConfig(**payload["model"])
    spec = AgentExecutionSpec(
        run_dir=run, thread_id=run.name, messages=payload.get("messages", []), model=model,
        selection=selection, retrieval=settings.retrieval, limits=limits,
        outputs=OutputContract(), interactive=False, mode="benchmark",
        context_window_tokens=settings.context_window_tokens,
        skills_root=None if preparing else Path(payload["template"]) / ".internal/agent_fs/skills",
    )
    result = await execute_agent(spec, prepare_only=preparing)
    if result["status"] != "completed":
        raise ConfigError(f"Capability execution failed: {result['stop_reason']}: {result.get('error_message', result.get('error_type', ''))}")
    result["resource_hash"] = json.loads((run / ".internal/manifest.json").read_text())["resource_hash"]
    atomic_write_json(run / "result.json", result)
    return result


def main(argv):
    payload = json.loads(Path(argv[0]).read_text())
    try:
        asyncio.run(execute(payload))
        return 0
    except (ConfigError, SecretsMissing, ValueError, OSError) as exc:
        atomic_write_json(Path(payload["reply_path"]), {"error": str(exc)})
        return 2
