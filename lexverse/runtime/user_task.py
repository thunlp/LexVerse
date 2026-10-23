from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

from lexverse.agents.general_runner import execute_general_agent
from lexverse.capabilities.preparation import prepare_agent_resources
from lexverse.runtime.paths import RunPaths, new_run_id
from lexverse.agents.spec import AgentExecutionSpec
from lexverse.capabilities.knowledge.loaders import load_documents, scan_source
from lexverse.config import ConfigError
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


def prepare_user_run(task: UserTaskSpec, output_root: Path, *, task_source: Path | None = None) -> Path:
    from lexverse.capabilities.registry import resolve_selection

    # Resolve capabilities without modifying the original task.
    task = task.model_copy(deep=True)
    resolve_selection(task.capabilities, task_source.parent if task_source else Path.cwd())

    # Create a unique run directory outside the input tree.
    run_id = new_run_id()
    run = output_root.resolve() / task.id / run_id
    run.mkdir(parents=True, exist_ok=False)
    if run.is_relative_to(task.inputs):
        run.rmdir()
        raise ValueError("Run directory must not be inside inputs")

    # Snapshot inputs and create the runtime and artifact directories.
    internal = run / ".internal"
    agent_fs = internal / "agent_fs"
    agent_fs.mkdir(parents=True)
    hashes = snapshot_inputs(task.inputs, agent_fs / "inputs")
    (run / "artifacts").mkdir()
    for name in ("workspace", "skills"):
        (agent_fs / name).mkdir()

    # Bind knowledge sources inside inputs to the saved snapshot.
    frozen = task.model_copy(deep=True)
    frozen.capabilities.knowledge = [
        agent_fs / "inputs" / source.relative_to(task.inputs) if source.is_relative_to(task.inputs) else source
        for source in task.capabilities.knowledge
    ]

    # Save the frozen task configuration and input hashes.
    atomic_write_json(internal / "manifest.json", {
        "schema_version": 1, "run_id": run_id, "task_id": task.id,
        "thread_id": run_id, "input_hashes": hashes,
        "task": frozen.model_dump(mode="json"),
        **({"task_source": str(task_source)} if task_source is not None else {}),
    })

    # Initialize the run result before resource preparation and execution.
    atomic_write_json(run / "result.json", {
        "status": "running", "stop_reason": None, "run_dir": str(run),
        "answer": None, "artifacts": [], "question": None, "usage": {},
        "trace_ref": str(internal / "events.jsonl"),
    })

    return run


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
        environment=task.environment,
        network_mode=task.network_mode, allowed_hosts=task.allowed_hosts,
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
        (paths.agent_fs / "inputs" / (relative + ".parsed.md")).write_text(content, encoding="utf-8")


async def _resume_agent(spec, *, reply, confirm_facts=False):
    paths = RunPaths(spec.run_dir)
    prior = json.loads((spec.run_dir / "result.json").read_text())
    spec.environment = prior.get("environment", spec.environment)
    manifest = json.loads((paths.internal / "manifest.json").read_text())
    resources = manifest.get("resources")
    try:
        if resources is None:
            raise ConfigError("Agent resources have not been prepared")
        return await execute_general_agent(
            spec, resources=resources, agent_name="lexverse",
            agent_kwargs={"reply": reply, "confirm_facts": confirm_facts},
        )
    except (ConfigError, ValueError) as exc:
        prior["resume_error"] = str(exc)
        atomic_write_json(spec.run_dir / "result.json", prior)
        return prior


async def _execute(args, task=None):
    if args.command == "run":
        # Use the final task configuration resolved by the CLI process.
        run = prepare_user_run(task, Path(args.output_root), task_source=Path(args.task))
        paths = RunPaths(run)

        # Parse input documents and save preparation failures.
        try:
            _parse_inputs(run)
        except (ConfigError, OSError, ValueError) as exc:
            result = {"status": "failed", "stop_reason": "input_preparation_failed", "run_dir": str(run),
                      "answer": None, "artifacts": [], "question": None, "error_type": type(exc).__name__,
                      "error_message": str(exc),
                      "trace_ref": str(paths.internal / "events.jsonl"), "usage": {
                          "model_calls": 0, "tool_calls": 0, "active_elapsed_sec": 0,
                          "input_tokens": 0, "output_tokens": 0, "usage_unavailable_calls": 0,
                          "human_calls": [], "model_breakdown": {}, "embedding": {}, "mcp": {},
                      }}
            manifest = json.loads((paths.internal / "manifest.json").read_text())
            with (paths.internal / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "event": "input_preparation_failed", "time": time.time(),
                    "run_id": manifest["run_id"], "task_id": manifest["task_id"],
                    "error_type": result["error_type"], "error_message": result["error_message"],
                }, ensure_ascii=False) + "\n")
            atomic_write_json(run / "result.json", result)
            return result

        # Build the execution specification from the saved manifest.
        spec = execution_spec(run)
        agent_name = task.agent

        # Prepare selected resources once before executing the agent.
        resources = await prepare_agent_resources(spec, agent_name=agent_name)

        # Execute the agent using the prepared resources.
        result = await execute_general_agent(
            spec,
            resources=resources, agent_name=agent_name,
        )
    else:
        # Resume an existing run with the supplied reply.
        run = Path(args.run_dir).expanduser().resolve()
        result = await _resume_agent(execution_spec(run), reply=args.reply, confirm_facts=args.confirm_facts)

    # Continue paused runs interactively when enabled.
    spec = execution_spec(run)
    while result["status"] == "needs_input" and spec.interactive and not result.get("resume_error"):
        # Display pending questions and factual summaries.
        questions = result["question"]
        for question in questions:
            print(question["question"], file=sys.stderr)
            if question["facts_summary"]:
                print(question["facts_summary"], file=sys.stderr)

        # Collect a reply and request factual confirmation when needed.
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
            # Leave the run paused when the user closes input.
            break

        # Submit the reply and continue from the saved checkpoint.
        result = await _resume_agent(spec, reply=reply, confirm_facts=confirm)

    return result


def main(payload):
    args = argparse.Namespace(**payload["args"])
    task = UserTaskSpec.model_validate(payload["task"]) if args.command == "run" else None
    try:
        result = asyncio.run(_execute(args, task))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"completed", "needs_input"} and not result.get("resume_error") else 1
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Task preparation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
