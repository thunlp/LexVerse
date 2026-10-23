from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import subprocess
import tempfile

from lexverse.runtime.paths import RunPaths
from lexverse.config import ConfigError
from lexverse.tasks.user import UserTaskSpec, load_user_task
from lexverse.runtime.dependencies import requires_agent_dependencies


def _worker_request(args):
    if args.command == "run":
        path = Path(args.task).expanduser().resolve()
        path = path / "task.yaml" if path.is_dir() else path
        task = load_user_task(path, Path(args.registry) if args.registry else None)
        task.agent = args.agent or task.agent
        task.environment = args.env or task.environment
        task.interactive = False if args.no_interactive else task.interactive
        if args.model:
            task.model["name"] = args.model
        required = requires_agent_dependencies(task.agent, task.capabilities.knowledge, task.inputs)
        return {"args": {"command": "run", "task": str(path), "output_root": args.output_root},
                "task": task.model_dump(mode="json"),
                "dependencies_required": required}

    run = Path(args.run_dir).expanduser().resolve()
    manifest = json.loads((RunPaths(run).internal / "manifest.json").read_text())
    prior = json.loads((run / "result.json").read_text())
    task = UserTaskSpec.model_validate(manifest["task"])
    agent = prior.get("agent", task.agent)
    if agent != "lexverse":
        raise ConfigError("Resume is supported only for the built-in lexverse agent")
    return {"args": {**vars(args), "run_dir": str(run)}, "dependencies_required": True}


def _run_worker(args):
    payload = _worker_request(args)
    with tempfile.TemporaryDirectory(prefix="lexverse-task-") as temporary:
        request = Path(temporary) / "request.json"
        request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return subprocess.call([sys.executable, "-m", "lexverse.runtime.worker", "task", str(request)])


def _prompt_yes_no(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def init_user_task(
    task_name: str,
    *,
    root: Path = Path("user_workspace/task"),
    docker_env: bool = True,
    oracle_solution: bool | None = None,
    verifier_script: bool | None = None,
) -> dict:
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", task_name):
        raise ConfigError(f"Invalid task name: {task_name!r}")
    task_dir = root.expanduser().resolve() / task_name
    if task_dir.exists() and any(task_dir.iterdir()):
        raise ConfigError(f"Task directory already exists and is not empty: {task_dir}")

    create_oracle = _prompt_yes_no("Create a reference solution for Oracle testing?") if oracle_solution is None else oracle_solution
    create_verifier = _prompt_yes_no("Create an automatic scoring script?") if verifier_script is None else verifier_script

    template_dir = Path(__file__).resolve().parents[2] / "user_workspace/task/example"
    (task_dir / "inputs").mkdir(parents=True, exist_ok=True)
    task_yaml = task_dir / "task.yaml"
    yaml_text = (template_dir / "task.yaml").read_text(encoding="utf-8")
    task_yaml.write_text(re.sub(r"^id:\s*\S+", f"id: {task_name}", yaml_text, count=1, flags=re.MULTILINE), encoding="utf-8")
    created_files = [str(task_yaml), str(task_dir / "inputs")]

    for enabled, rel in (
        (docker_env, "environment/Dockerfile"),
        (create_oracle, "solution/solve.sh"),
        (create_verifier, "tests/test.sh"),
    ):
        if enabled:
            dest = task_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template_dir / rel, dest)
            created_files.append(str(dest))

    return {
        "task_id": task_name,
        "task_dir": str(task_dir),
        "task_yaml": str(task_yaml),
        "created": created_files,
    }


def summarize_atif_trajectory(trajectory_path: Path) -> dict:
    """
    Validate and summarize an ATIF-v1.7 `trajectory.json` file, returning step counts,
    tool call breakdown, and token/cost metrics.
    """
    from lexverse.models.trajectories import Trajectory

    traj = Trajectory.model_validate_json(trajectory_path.read_text(encoding="utf-8"))
    source_counts = {"system": 0, "user": 0, "agent": 0}
    tool_counts: dict[str, int] = {}
    total_tool_calls = 0

    for step in traj.steps:
        source_counts[step.source] = source_counts.get(step.source, 0) + 1
        if step.tool_calls:
            for tc in step.tool_calls:
                total_tool_calls += 1
                tool_counts[tc.function_name] = tool_counts.get(tc.function_name, 0) + 1

    return {
        "schema_version": traj.schema_version,
        "session_id": traj.session_id,
        "agent": traj.agent.model_dump(exclude_none=True),
        "total_steps": len(traj.steps),
        "steps_by_source": source_counts,
        "total_tool_calls": total_tool_calls,
        "tool_calls_by_name": tool_counts,
        "final_metrics": traj.final_metrics.model_dump(exclude_none=True) if traj.final_metrics else None,
    }


def main(argv=None):
    # Set up the task command group.
    parser = argparse.ArgumentParser(prog="lexverse task")
    commands = parser.add_subparsers(dest="command", required=True)

    # Define task initialization options.
    init_cmd = commands.add_parser("init", help="initialize a new user task directory under user_workspace/task/<task_name>/")
    init_cmd.add_argument("task_name", help="Name of the task directory to create")
    init_cmd.add_argument("--root", default="user_workspace/task", help="Parent directory for user tasks (default: user_workspace/task)")

    # Define execution options and configuration overrides.
    run = commands.add_parser("run", help="run a legal task locally or in a Docker sandbox")
    run.add_argument("task")
    run.add_argument("--registry")
    run.add_argument("--output-root", default="runs/user_tasks")
    run.add_argument("--no-interactive", action="store_true")
    run.add_argument("--agent", default=None, help="Agent name (e.g. lexverse, claude-code, codex, qwen-coder, gemini-cli, openhands, opencode, aider, oracle) or module:Class")
    run.add_argument("--env", default=None, help="Execution environment type (local or docker)")
    run.add_argument("--model", default=None, help="Optional model name override")

    # Define resume, inspection, and agent discovery options.
    resume = commands.add_parser("resume", help="resume a paused interactive task run")
    resume.add_argument("run_dir")
    resume.add_argument("--reply", required=True)
    resume.add_argument("--confirm-facts", action="store_true")
    inspect = commands.add_parser("inspect", help="inspect a completed or paused task run")
    inspect.add_argument("run_dir")
    inspect.add_argument("--trajectory", action="store_true", help="include ATIF trajectory summary from trajectory.json")
    commands.add_parser("agents", help="list all supported built-in and general CLI agents")
    args = parser.parse_args(argv)

    try:
        # Handle initialization and agent discovery in the CLI process.
        if args.command == "init":
            payload = init_user_task(args.task_name, root=Path(args.root))
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "agents":
            from lexverse.agents.factory import AgentFactory
            payload = {
                "default_agent": "lexverse",
                "supported_environments": ["local", "docker"],
                "supported_agents": AgentFactory.supported_agent_names(),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect":
            # Read saved results without starting an execution worker.
            run_dir = Path(args.run_dir)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            traj_file = run_dir / "trajectory.json"
            if getattr(args, "trajectory", False) and traj_file.is_file():
                result["trajectory_summary"] = summarize_atif_trajectory(traj_file)
        else:
            # Delegate run and resume commands to the runtime worker.
            return _run_worker(args)

        # Display the inspected result and return its exit status.
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"completed", "needs_input"} and not result.get("resume_error") else 1
    except (ConfigError, OSError, ValueError) as exc:
        # Report expected task errors with a consistent exit code.
        print(f"Task preparation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
