from pathlib import Path


def execution_identity(spec, *, agent_name="lexverse"):
    if agent_name != "lexverse":
        return {
            "agent": agent_name, "environment": spec.environment, "mode": spec.mode,
            "selection": spec.selection.model_dump(mode="json"),
            "network_mode": spec.network_mode, "allowed_hosts": spec.allowed_hosts,
        }

    from dataclasses import asdict
    import hashlib
    import importlib.metadata
    import subprocess
    from lexverse.capabilities.base import identity
    from lexverse.capabilities.skills import tree_hashes
    from lexverse.runtime.paths import RunPaths

    package = Path(__file__).resolve().parents[1]
    files = [*package.joinpath("agents").glob("*.py"), *package.joinpath("capabilities").rglob("*.py"),
             package / "commands/tasks.py", package / "tasks/user.py", package / "providers/api.py",
             package / "runtime/paths.py", package / "runtime/enhancement.py",
             package / "runtime/identity.py", package / "interaction/participants.py", package / "interaction/engine.py", package / "config.py"]
    files += [package / "runtime" / name for name in (
        "dependencies.py", "worker.py", "user_task.py",
    )]
    files += [*package.joinpath("environments").glob("*.py"),
              *(path for path in package.joinpath("environments/egress").rglob("*") if path.is_file())]
    versions = {name: importlib.metadata.version(name) for name in (
        "deepagents", "langchain", "langchain-core", "langchain-openai", "langgraph",
        "langgraph-checkpoint", "langgraph-checkpoint-sqlite", "fastmcp", "pydantic",
    )}
    try:
        revision = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=5, check=False)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {"model": asdict(spec.model), "messages_hash": identity(spec.messages), "mode": spec.mode,
            "selection": spec.selection.model_dump(mode="json"), "retrieval": spec.retrieval.model_dump(),
            "limits": spec.limits.model_dump(), "outputs": spec.outputs.model_dump(),
            "context_window_tokens": spec.context_window_tokens,
            "network_mode": spec.network_mode, "allowed_hosts": spec.allowed_hosts,
            "code_commit": commit,
            "code": {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "versions": versions, "input_files": tree_hashes(RunPaths(spec.run_dir).agent_fs / "inputs")}

