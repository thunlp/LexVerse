import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from lexverse.config import ConfigError
from lexverse.capabilities.base import execution_conditions
from lexverse.runtime.results import atomic_write_json


def enabled(config):
    return config is not None and config.generation.get("capabilities", {}).get("enabled", False)


async def run_worker(payload, control_dir):
    control_dir.mkdir(parents=True, exist_ok=True)
    request = control_dir / "request.json"
    error = control_dir / "error.json"
    payload = {**payload, "reply_path": str(error)}
    error.unlink(missing_ok=True)
    atomic_write_json(request, payload)
    with (control_dir / "worker.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "lexverse.agents.entrypoint", "benchmark", str(request),
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
        message = json.loads(error.read_text())["error"] if error.exists() else f"Stage2 worker failed; see {control_dir / 'worker.log'}"
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
