from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from lexverse.agents.general_runner import execute_general_agent
from lexverse.runtime.paths import RunPaths
from lexverse.capabilities.base import AgentExecutionSpec
from lexverse.capabilities.knowledge.loaders import load_documents, scan_source
from lexverse.config import ConfigError
from lexverse.environments.user_task import prepare_user_run
from lexverse.runtime.results import atomic_write_json
from lexverse.tasks.user import UserTaskSpec, load_user_task


def execution_spec(run: Path) -> AgentExecutionSpec:
    paths = RunPaths(run)
    manifest = json.loads((paths.internal / "manifest.json").read_text())
    task = UserTaskSpec.model_validate(manifest["task"])
    listing = sorted("/" + path.relative_to(paths.agent_fs).as_posix() for path in (paths.agent_fs / "inputs").rglob("*") if path.is_file())
    instruction = task.instruction + "\n\nInput files: " + json.dumps(listing, ensure_ascii=False)
    instruction += "\nRequired output files: " + json.dumps(["/" + path for path in task.outputs.required])
    return AgentExecutionSpec(
        run_dir=run, thread_id=manifest["thread_id"], messages=[{"role": "user", "content": instruction}],
        model=task.model_config_value(Path(manifest.get("task_source", run)).parent), selection=task.capabilities,
        retrieval=task.retrieval, limits=task.limits, outputs=task.outputs, interactive=task.interactive,
        context_window_tokens=task.context_window_tokens,
    )


def _parse_inputs(run: Path):
    paths = RunPaths(run)
    files, skipped = scan_source(paths.inputs)
    unsupported = [item for item in skipped if item["reason"] == "unsupported_format"]
    if unsupported:
        raise ConfigError("Unsupported input formats: " + ", ".join(item["file"] for item in unsupported))
    for relative in files:
        if Path(relative).suffix.lower() not in {".pdf", ".docx", ".xlsx", ".pptx"}:
            continue
        documents = list(load_documents(paths.inputs, {relative: files[relative]}, "inputs", use_manifest=False))
        content = "\n\n".join(f"[{doc.metadata['position']}]\n{doc.page_content}" for doc in documents)
        target = paths.agent_fs / "inputs" / (relative + ".parsed.md")
        target.write_text(content, encoding="utf-8")
        top_target = paths.inputs / (relative + ".parsed.md")
        if top_target.resolve() != target.resolve():
            top_target.write_text(content, encoding="utf-8")


async def _execute(args):
    if args.command == "run":
        task_path = Path(args.task).expanduser().resolve()
        task = load_user_task(task_path, Path(args.registry) if args.registry else None)
        if args.no_interactive:
            task.interactive = False
        run = prepare_user_run(task, Path(args.output_root))
        paths = RunPaths(run)
        manifest = json.loads((paths.internal / "manifest.json").read_text())
        manifest["task_source"] = str(task_path)
        atomic_write_json(paths.internal / "manifest.json", manifest)
        try:
            _parse_inputs(run)
        except (ConfigError, OSError, ValueError) as exc:
            result = {"status": "failed", "stop_reason": "input_preparation_failed", "run_dir": str(run),
                      "answer": None, "artifacts": [], "question": None, "error_type": type(exc).__name__,
                      "error_message": str(exc),
                      "trace_ref": str(paths.internal / "events.jsonl"), "usage": {}}
            atomic_write_json(run / "result.json", result)
            return result
        agent_name = getattr(args, "agent", None) or task.agent or "lexverse"
        env_type = getattr(args, "env", None) or task.environment or "docker"
        override_model = getattr(args, "model", None)
        if agent_name in ("lexverse", "deepagents"):
            from lexverse.agents.agent_loop import execute_agent

            result = await execute_agent(execution_spec(run))
        else:
            result = await execute_general_agent(
                execution_spec(run),
                agent_name=agent_name,
                env_type=env_type,
                override_model_name=override_model,
            )
    else:
        from lexverse.agents.agent_loop import execute_agent

        run = Path(args.run_dir).expanduser().resolve()
        result = await execute_agent(execution_spec(run), reply=args.reply, confirm_facts=args.confirm_facts)
    spec = execution_spec(run)
    while result["status"] == "needs_input" and spec.interactive and not result.get("resume_error"):
        questions = result["question"]
        for question in questions:
            print(question["question"], file=sys.stderr)
            if question["facts_summary"]:
                print(question["facts_summary"], file=sys.stderr)
        try:
            confirm = False
            while True:
                reply = input("Your reply (Ctrl-D to leave paused): ")
                if len(questions) != 1 or not questions[0]["summary_hash"]:
                    break
                while True:
                    decision = input("Confirm displayed facts? [yes/no]: ").strip().lower()
                    if decision in {"y", "yes", "n", "no"}:
                        break
                    print("Please enter yes or no.", file=sys.stderr)
                if decision in {"y", "yes"}:
                    confirm = True
                    break
                print("Reply not submitted. Please re-enter.", file=sys.stderr)
        except EOFError:
            break
        result = await execute_agent(spec, reply=reply, confirm_facts=confirm)
    return result


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
    parser = argparse.ArgumentParser(prog="lexverse task")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a legal task in a local or docker sandbox")
    run.add_argument("task")
    run.add_argument("--registry")
    run.add_argument("--output-root", default="runs/user_tasks")
    run.add_argument("--no-interactive", action="store_true")
    run.add_argument("--agent", default=None, help="Agent name (e.g. lexverse, claude-code, codex, qwen-code, gemini-cli, openhands, opencode, aider, oracle) or module:Class")
    run.add_argument("--env", default=None, help="Execution environment type (local or docker)")
    run.add_argument("--model", default=None, help="Optional model name override for external CLI agents (provider/model_name)")
    resume = commands.add_parser("resume", help="resume a paused interactive task run")
    resume.add_argument("run_dir")
    resume.add_argument("--reply", required=True)
    resume.add_argument("--confirm-facts", action="store_true")
    inspect = commands.add_parser("inspect", help="inspect a completed or paused task run")
    inspect.add_argument("run_dir")
    inspect.add_argument("--trajectory", action="store_true", help="include ATIF trajectory summary from trajectory.json")
    agents_cmd = commands.add_parser("agents", help="list all supported built-in and general CLI agents")
    args = parser.parse_args(argv)
    try:
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
            run_dir = Path(args.run_dir)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            traj_file = run_dir / "trajectory.json"
            if getattr(args, "trajectory", False) and traj_file.is_file():
                result["trajectory_summary"] = summarize_atif_trajectory(traj_file)
        else:
            result = asyncio.run(_execute(args))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"completed", "needs_input"} and not result.get("resume_error") else 1
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Task preparation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
