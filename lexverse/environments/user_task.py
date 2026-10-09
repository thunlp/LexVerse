from datetime import datetime, timezone
import hashlib
from pathlib import Path
from uuid import uuid4

from lexverse.runtime.results import atomic_write_json
from lexverse.tasks.user import UserTaskSpec


def snapshot_inputs(source: Path, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Input symbolic links are not supported: {path.relative_to(source)}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            content = path.read_bytes()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
        else:
            raise ValueError(f"Unsupported input entry: {relative}")
    return hashes


def prepare_user_run(task: UserTaskSpec, output_root: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    run = output_root.resolve() / task.id / run_id
    run.mkdir(parents=True, exist_ok=False)
    if run.is_relative_to(task.inputs):
        run.rmdir()
        raise ValueError("Run directory must not be inside inputs")
    internal = run / ".internal"
    agent_fs = internal / "agent_fs"
    agent_fs.mkdir(parents=True)
    hashes = snapshot_inputs(task.inputs, agent_fs / "inputs")
    (run / "artifacts").mkdir()
    for name in ("workspace", "skills"):
        (agent_fs / name).mkdir()
    frozen = task.model_copy(deep=True)
    frozen.capabilities.knowledge = [
        agent_fs / "inputs" / source.relative_to(task.inputs) if source.is_relative_to(task.inputs) else source
        for source in task.capabilities.knowledge
    ]
    atomic_write_json(internal / "manifest.json", {
        "schema_version": 1, "run_id": run_id, "task_id": task.id,
        "thread_id": run_id, "input_hashes": hashes,
        "task": frozen.model_dump(mode="json"),
    })
    atomic_write_json(run / "result.json", {
        "status": "running", "stop_reason": None, "run_dir": str(run),
        "answer": None, "artifacts": [], "question": None, "usage": {},
        "trace_ref": str(internal / "events.jsonl"),
    })
    return run
