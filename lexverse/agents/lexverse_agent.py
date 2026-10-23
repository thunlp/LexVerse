from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, override

from filelock import FileLock
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from lexverse.agents.atif_converter import convert_langchain_messages_to_atif, write_atif_trajectory
from lexverse.agents.factory import build_agent
from lexverse.agents.base import BaseAgent
from lexverse.agents.general_runner import collect_artifacts as _artifacts
from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.runtime.paths import CAPABILITIES_ROOT, RunPaths
from lexverse.agents.spec import AgentExecutionSpec
from lexverse.runtime.identity import execution_identity as _execution_identity
from lexverse.capabilities.base import execution_conditions
from lexverse.capabilities.preparation import load_agent_resources
from lexverse.capabilities.middleware import BudgetExceeded, DomainMiddleware, RunLedger
from lexverse.capabilities.tools import ask_user
from lexverse.config import ConfigError
from lexverse.providers.api import create_chat_model
from lexverse.runtime.results import atomic_write_json


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _pending(snapshot):
    interrupts = [interrupt for task in snapshot.tasks for interrupt in task.interrupts]
    if not interrupts:
        return []
    actions = []
    for interrupt in interrupts:
        if not isinstance(interrupt.value, dict) or "action_requests" not in interrupt.value:
            raise ConfigError("Unsupported interrupt payload; cannot synthesize a user reply")
        for index, action in enumerate(interrupt.value["action_requests"]):
            if action["name"] != "ask_user":
                raise ConfigError("Unexpected approval request")
            arguments = action["args"]
            summary = arguments.get("facts_summary")
            actions.append({"interrupt_id": interrupt.id, "index": index, "question": arguments["question"],
                            "facts_summary": summary, "summary_hash": _identity(summary) if summary else None})
    return actions


async def execute_agent(spec: AgentExecutionSpec, *, system_prompt: str, environment: ExecutionEnvironment,
                        reply=None, confirm_facts=False,
                        model=None, cache: Path | None = None, adapter_factory=None,
                        resources: dict | None = None) -> dict:
    # Initialize run paths, usage tracking, and the default result.
    run = spec.run_dir.resolve()
    cache = cache or CAPABILITIES_ROOT
    paths = RunPaths(run)
    ledger = RunLedger(run, spec.limits)
    result = {"status": "failed", "stop_reason": "preparation_failed", "run_dir": str(run),
              "answer": None, "artifacts": [], "question": None, "trace_ref": str(paths.internal / "events.jsonl")}

    # Prevent concurrent execution of the same run.
    with FileLock(str(paths.internal / "execution.lock"), timeout=0):
        try:
            manifest = _read(paths.internal / "manifest.json")
            resuming = reply is not None
            if resuming:
                prior = _read(run / "result.json")
                if prior["status"] != "needs_input":
                    raise ConfigError("Only a paused run can accept a reply")

            # Require prepared resources and verify execution compatibility.
            saved = resources if resources is not None else manifest.get("resources")
            if saved is None:
                raise ConfigError("Agent resources have not been prepared")
            execution_identity = _execution_identity(spec)
            if manifest.get("execution") and execution_conditions(manifest["execution"]) != execution_conditions(execution_identity):
                raise ConfigError("Execution configuration, dependencies or inputs changed; start a new run")
            if manifest.get("execution"):
                previous = manifest.get("execution_history", [manifest["execution"]])[-1]
                if previous != execution_identity:
                    manifest.setdefault("execution_history", []).append(execution_identity)
                    atomic_write_json(paths.internal / "manifest.json", manifest)
            if not manifest.get("execution"):
                manifest["execution"] = execution_identity
                atomic_write_json(paths.internal / "manifest.json", manifest)

            skills = saved["skills"]
            async with AsyncExitStack() as stack:
                if spec.network_mode == "allowlist" and (
                    environment.type().value != "docker" or environment.task_env_config.network_mode != "allowlist"
                ):
                    raise ConfigError("The supplied environment does not enforce the task network allowlist")
                await stack.enter_async_context(environment.agent_execution())
                # Open MCP connections and register local tools.
                tools = await stack.enter_async_context(load_agent_resources(
                    spec, saved, cache=cache, adapter_factory=adapter_factory,
                    embedding_recorder=ledger.record_embedding,
                ))
                if "ask_user" in spec.selection.tools:
                    tools.append(ask_user)

                # Build the model and agent graph with persistent checkpoints.
                chat_model = model if model is not None else create_chat_model(
                    spec.model, context_window_tokens=spec.context_window_tokens,
                )
                saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(paths.internal / "checkpoint.sqlite")))
                graph = build_agent(chat_model, environment, tools=tools, skills=skills["paths"],
                                    timeout_sec=spec.limits.tool_timeout_sec,
                                    checkpointer=saver, system_prompt=system_prompt,
                                    middleware=[DomainMiddleware(ledger, skills=skills["skills"])])

                # Start with task messages, or resume a pending human reply.
                config = {"configurable": {"thread_id": spec.thread_id}, "callbacks": [ledger],
                          "recursion_limit": 4 * (spec.limits.model_calls + spec.limits.tool_calls) + 20}
                initial = {"messages": deepcopy(spec.messages), "facts_confirmation": {"confirmed": False}}
                if resuming:
                    snapshot = await graph.aget_state(config)
                    questions = _pending(snapshot)
                    if not questions or questions != prior["question"]:
                        raise ConfigError("Pending question does not match checkpoint")
                    if confirm_facts and (len(questions) != 1 or not questions[0]["summary_hash"]):
                        raise ConfigError("Explicit confirmation requires one displayed factual summary")
                    confirmation = {"confirmed": confirm_facts, "summary_hash": questions[0]["summary_hash"] if confirm_facts else None,
                                    "reply": reply, "confirmed_at": time.time() if confirm_facts else None}
                    grouped = {}
                    for question in questions:
                        grouped.setdefault(question["interrupt_id"], {"decisions": []})["decisions"].append({"type": "respond", "message": reply})
                    initial = Command(resume=grouped, update={"facts_confirmation": confirmation})
                    ledger.event("human_resumed", questions=questions, confirmation=confirmation)

                # Start timing only after runtime setup and resume checks.
                if paths.compact:
                    ledger.update_result(status="running", stop_reason=None, question=None)
                ledger.begin()
                ledger.event("run_started" if not resuming else "run_resumed")
                messages_for_atif = []
                # Allow one extra graph invocation to repair missing outputs.
                for attempt in range(2):
                    remaining = ledger.remaining_time()
                    if remaining <= 0:
                        raise BudgetExceeded("active_timeout")
                    # Run the model/tool loop within the remaining time budget.
                    await asyncio.wait_for(graph.ainvoke(initial, config=config), timeout=remaining)
                    snapshot = await graph.aget_state(config)
                    messages_for_atif = snapshot.values.get("messages", [])
                    questions = _pending(snapshot)

                    # Save pending questions and return control to the caller.
                    if questions:
                        for question in questions:
                            ledger.reserve_question(question["interrupt_id"] + ":" + str(question["index"]))
                        result.update(status="needs_input", stop_reason="ask_user", question=questions)
                        ledger.event("human_interrupt", questions=questions)
                        break

                    # Extract and validate the final model answer.
                    messages = messages_for_atif
                    final = messages[-1] if messages else None
                    if not isinstance(final, AIMessage) or final.tool_calls or not final.content:
                        raise ValueError("Agent did not produce a final answer")
                    answer = final.content if isinstance(final.content, str) else "\n".join(
                        block["text"] for block in final.content if isinstance(block, dict) and block.get("type") == "text"
                    )
                    answer = re.sub(r"(?s)<think>.*?</think>\s*", "", answer)
                    if "</think>" in answer:
                        answer = answer.split("</think>", 1)[1].lstrip()
                    if not answer.strip():
                        raise ValueError("Agent final answer is empty")

                    # Require valid output files before marking completion.
                    artifacts, missing = _artifacts(spec)
                    if missing and attempt == 0:
                        initial = {"messages": [HumanMessage(content="Required output files are missing or invalid: " + ", ".join(missing) + ". Create or repair them before finishing.")]}
                        ledger.event("output_repair", missing=missing)
                        continue
                    if missing:
                        result.update(stop_reason="output_missing", artifacts=artifacts)
                    else:
                        result.update(
                            status="completed",
                            stop_reason="final_answer",
                            answer=answer,
                            artifacts=artifacts,
                        )
                    break

                # Export the message trajectory when available.
                if messages_for_atif:
                    try:
                        atif_traj = convert_langchain_messages_to_atif(
                            messages_for_atif,
                            session_id=spec.thread_id,
                            agent_name="lexverse",
                            model_name=getattr(spec.model, "model", None) or getattr(spec.model, "name", None),
                            usage=ledger.usage,
                        )
                        if atif_traj is not None:
                            write_atif_trajectory(atif_traj, run / "trajectory.json")
                            result["trajectory_ref"] = str(run / "trajectory.json")
                    except Exception:
                        pass
        # Translate execution failures into result statuses.
        except (BudgetExceeded, asyncio.TimeoutError) as exc:
            result.update(status="limit_reached", stop_reason=str(exc) if isinstance(exc, BudgetExceeded) else "active_timeout")
        except asyncio.CancelledError:
            result.update(status="cancelled", stop_reason="cancelled")
            raise
        except ConfigError as exc:
            if resuming and ledger.started is None:
                result = dict(prior)
                result["resume_error"] = str(exc)
            else:
                result.update(error_type=type(exc).__name__, error_message=str(exc))
            ledger.event("run_failed", error_type=type(exc).__name__)
        except Exception as exc:
            result.update(stop_reason="execution_failed" if ledger.started is not None else "preparation_failed", error_type=type(exc).__name__)
            ledger.event("run_failed", error_type=type(exc).__name__)
        finally:
            # Persist usage and retain partial artifacts for unfinished runs.
            if not result.get("resume_error"):
                ledger.finish()
                if result["status"] != "completed":
                    result["artifacts"], _ = _artifacts(spec, strict=False)
                    for artifact in result["artifacts"]:
                        artifact["partial"] = True
                result["usage"] = ledger.usage
                if not paths.compact:
                    atomic_write_json(run / "artifacts.json", result["artifacts"])
            atomic_write_json(run / "result.json", result)
            ledger.event("run_finished", status=result["status"], stop_reason=result["stop_reason"])
    return result


class LexVerseAgent(BaseAgent):
    """
    Built-in LexVerse LangGraph/DeepAgents implementation wrapped as a standard
    Harbor-compatible BaseAgent.
    """

    SUPPORTS_ATIF: bool = True
    SUPPORTS_RESUME: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        spec: AgentExecutionSpec | None = None,
        reply: str | None = None,
        confirm_facts: bool = False,
        model: Any | None = None,
        cache: Path | None = None,
        adapter_factory: Any | None = None,
        resources: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, *args, **kwargs)
        self.spec = spec
        self.reply = reply
        self.confirm_facts = confirm_facts
        self.model = model
        self.cache = cache
        self.adapter_factory = adapter_factory
        self.resources = resources
        self.last_result: dict[str, Any] | None = None

    @staticmethod
    @override
    def name() -> str:
        return AgentName.LEXVERSE.value

    @override
    def version(self) -> str:
        return "0.2.0"

    @override
    async def setup(self, environment: ExecutionEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: ExecutionEnvironment,
        context: AgentContext,
    ) -> None:
        if self.spec is None:
            raise ValueError("LexVerseAgent requires an AgentExecutionSpec (`spec`).")

        result = await execute_agent(
            self.spec,
            system_prompt=instruction,
            reply=self.reply,
            confirm_facts=self.confirm_facts,
            model=self.model,
            cache=self.cache,
            adapter_factory=self.adapter_factory,
            resources=self.resources,
            environment=environment,
        )
        self.last_result = result
        usage = result.get("usage") or {}
        context.n_input_tokens = usage.get("input_tokens")
        context.n_output_tokens = usage.get("output_tokens")
        context.metadata = {"lexverse_result": result}

        # Also copy ATIF trajectory.json into logs_dir if logs_dir != run_dir
        traj_src = self.spec.run_dir.resolve() / "trajectory.json"
        traj_dst = self.logs_dir / "trajectory.json"
        if traj_src.exists() and traj_src.resolve() != traj_dst.resolve():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            traj_dst.write_text(traj_src.read_text(encoding="utf-8"), encoding="utf-8")
