from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
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
from lexverse.agents.model_connection import ModelConnectionSpec
from lexverse.agents.base import BaseAgent
from lexverse.trial.artifacts import collect as collect_outputs
from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.utils.paths import CAPABILITIES_ROOT
from lexverse.models.trial.paths import RunPaths
from lexverse.agents.base import AgentExecutionSpec
from lexverse.agents.base import execution_identity as _execution_identity
from lexverse.capabilities.base import execution_conditions
from lexverse.capabilities.preparation import load_agent_resources
from lexverse.capabilities.middleware import BudgetExceeded, DomainMiddleware, RunLedger
from lexverse.capabilities.tools import ask_user
from lexverse.config import ConfigError
from lexverse.llms.api import create_chat_model
from lexverse.utils.files import atomic_write_json
from lexverse.utils.identity import identity as _identity


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


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

    messages_for_atif = []
    result.update(trajectory_status="unavailable", trajectory_error="No messages available for ATIF conversion")

    def save_trajectory():
        if not messages_for_atif:
            return
        try:
            trajectory = convert_langchain_messages_to_atif(
                messages_for_atif, session_id=spec.thread_id, agent_name="vanilla",
                model_name=spec.model.name, usage=ledger.usage,
            )
            if trajectory is not None:
                write_atif_trajectory(trajectory, run / "trajectory.json")
                result.update(trajectory_ref=str(run / "trajectory.json"), trajectory_status="saved")
                result.pop("trajectory_error", None)
        except Exception as exc:
            result.update(trajectory_status="failed", trajectory_error=type(exc).__name__)
            ledger.event("trajectory_failed", error_type=type(exc).__name__)

    # Prevent concurrent execution of the same run.
    with FileLock(str(paths.internal / "execution.lock"), timeout=0):
        try:
            manifest = _read(paths.internal / "manifest.json")
            resuming = reply is not None
            if resuming:
                prior = _read(RunPaths(run).worker_result)
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
                    embedding_recorder=ledger.record_embedding, environment=environment,
                ))
                if spec.mode != "benchmark" and "ask_user" in spec.selection.tools:
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
                ledger.update_result(status="running", stop_reason=None, question=None)
                ledger.begin()
                ledger.event("run_started" if not resuming else "run_resumed")
                async def invoke():
                    nonlocal messages_for_atif
                    async for values in graph.astream(initial, config=config, stream_mode="values"):
                        messages_for_atif = values.get("messages", [])
                        save_trajectory()
                # Allow one extra graph invocation to repair missing outputs.
                for attempt in range(2):
                    remaining = ledger.remaining_time()
                    if remaining <= 0:
                        raise BudgetExceeded("active_timeout")
                    # Run the model/tool loop within the remaining time budget.
                    await asyncio.wait_for(invoke(), timeout=remaining)
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
                    _, artifacts, missing = await collect_outputs(environment, spec.artifacts, spec.run_dir / "artifacts")
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
                save_trajectory()
                if result["status"] != "completed":
                    _, result["artifacts"], _ = await collect_outputs(environment, spec.artifacts, spec.run_dir / "artifacts")
                    for artifact in result["artifacts"]:
                        artifact["partial"] = True
                result["usage"] = ledger.usage
            atomic_write_json(RunPaths(run).worker_result, result)
            ledger.event("run_finished", status=result["status"], stop_reason=result["stop_reason"])
    return result


class VanillaAgent(BaseAgent):
    SUPPORTS_NETWORK_ALLOWLIST = True
    MODEL_CONNECTION = ModelConnectionSpec(prefixed_model=False)
    
    SUPPORTS_SKILLS = True
    MCP_TRANSPORTS = frozenset({"stdio", "streamable-http", "sse"})
    SUPPORTS_MCP_HEADERS = True
    SUPPORTS_MCP_ENV = True
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

    def read_final_answer(self) -> str | None:
        return (self.last_result or {}).get("answer")

    @staticmethod
    @override
    def name() -> str:
        return AgentName.VANILLA.value

    @override
    def version(self) -> str:
        return "1.0"

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
            raise ValueError("VanillaAgent requires an AgentExecutionSpec (`spec`).")

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


def build_agent(model, environment, *, tools=(), skills=None, checkpointer=None,
                middleware=(), interactive=False, system_prompt="", timeout_sec: float = 60):
    from deepagents import create_deep_agent
    from lexverse.agents.environment_backend import EnvironmentBackend
    from deepagents.profiles import (
        GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile,
    )
    from pydantic import ValidationError

    provider = model._get_ls_params().get("ls_provider")
    identifier = getattr(model, "model_name", None) or getattr(model, "model", None)
    if not provider:
        raise ValueError("Chat model must identify its provider for single-agent configuration")
    key = f"{provider}:{identifier}" if identifier else provider
    register_harness_profile(key, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"task", "delete"}),
    ))
    backend = EnvironmentBackend(environment, timeout_sec=timeout_sec)
    user_tool = next((tool for tool in tools if getattr(tool, "name", None) == "ask_user"), None)

    def valid_question(request):
        try:
            arguments = user_tool.args_schema.model_validate(request.tool_call["args"])
            return bool(arguments.question.strip())
        except ValidationError:
            return False

    return create_deep_agent(
        model=model, backend=backend,
        tools=list(tools), skills=skills, middleware=list(middleware),
        checkpointer=checkpointer, system_prompt=system_prompt,
        interrupt_on={"ask_user": {"allowed_decisions": ["respond"], "when": valid_question}}
        if user_tool is not None else None,
    )
