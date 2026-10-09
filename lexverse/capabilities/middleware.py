from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphInterrupt

from lexverse.runtime.results import atomic_write_json
from lexverse.runtime.paths import RunPaths
from lexverse.tasks.user import AgentLimits
from lexverse.capabilities.mcp import failure_category, response_failed


class BudgetExceeded(RuntimeError):
    pass


class ExecutionState(AgentState):
    facts_confirmation: dict


class RunLedger(BaseCallbackHandler):
    raise_error = True

    def __init__(self, run: Path, limits: AgentLimits):
        self.run, self.limits = run, limits
        self.paths = RunPaths(run)
        manifest_path = self.paths.internal / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        self.identity = {key: manifest[key] for key in ("run_id", "task_id", "participant_id", "attempt_id") if key in manifest}
        self.lock = threading.RLock()
        path = run / "usage.json"
        if self.paths.compact:
            path = run / "result.json"
            usage = json.loads(path.read_text()).get("usage", {}) if path.exists() else {}
        else:
            usage = json.loads(path.read_text()) if path.exists() else {}
        self.usage = usage or {
            "model_calls": 0, "tool_calls": 0, "active_elapsed_sec": 0,
            "input_tokens": 0, "output_tokens": 0, "usage_unavailable_calls": 0,
            "human_calls": [],
        }
        self.started = None
        self.model_kinds = {}
        self.usage.setdefault("model_breakdown", {})
        self.usage.setdefault("embedding", {})
        self.usage.setdefault("mcp", {})

    def record_embedding(self, kind, texts, tokens, status):
        with self.lock:
            counts = self.usage["embedding"].setdefault(kind, {"calls": 0, "texts": 0, "input_tokens": 0, "failed_calls": 0})
            counts["calls"] += 1
            counts["texts"] += texts
            counts["input_tokens"] += tokens
            counts["failed_calls"] += status == "failed"
            self.save()
        self.event("embedding_completed" if status == "completed" else "embedding_failed",
                   purpose=kind, texts=texts, input_tokens=tokens)

    def record_mcp(self, service, status):
        with self.lock:
            counts = self.usage["mcp"].setdefault(service, {"calls": 0, "completed_calls": 0, "failed_calls": 0})
            if status == "started":
                counts["calls"] += 1
            elif status == "error":
                counts["failed_calls"] += 1
            else:
                counts["completed_calls"] += 1
            self.save()

    def event(self, event: str, **fields):
        with self.lock:
            with (self.paths.internal / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": event, "time": time.time(), **self.identity, **fields}, ensure_ascii=False, default=str) + "\n")

    def update_result(self, **fields):
        with self.lock:
            path = self.run / "result.json"
            result = json.loads(path.read_text()) if path.exists() else {"status": "running"}
            result.update(fields)
            result["usage"] = self.usage
            atomic_write_json(path, result)

    def save(self):
        if self.paths.compact:
            self.update_result()
        else:
            atomic_write_json(self.run / "usage.json", self.usage)

    def remaining_time(self):
        current = time.monotonic() - self.started if self.started is not None else 0
        return self.limits.active_timeout_sec - self.usage["active_elapsed_sec"] - current

    def reserve(self, kind: str):
        with self.lock:
            field = kind + "_calls"
            if self.remaining_time() <= 0 or self.usage[field] >= getattr(self.limits, field):
                raise BudgetExceeded(field if self.remaining_time() > 0 else "active_timeout")
            self.usage[field] += 1
            self.save()

    def reserve_question(self, call_id: str):
        with self.lock:
            if call_id not in self.usage["human_calls"]:
                self.reserve("tool")
                self.usage["human_calls"].append(call_id)
                self.save()

    def begin(self):
        self.started = time.monotonic()

    def finish(self):
        if self.started is not None:
            self.usage["active_elapsed_sec"] += time.monotonic() - self.started
            self.started = None
        self.save()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.reserve("model")
        kind = "summary" if (kwargs.get("metadata") or {}).get("lc_source") == "summarization" else "decision"
        with self.lock:
            self.model_kinds[str(run_id)] = kind
            counts = self.usage["model_breakdown"].setdefault(kind, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "usage_unavailable_calls": 0,
            })
            counts["calls"] += 1
            self.save()
        self.event("model_started", call_id=str(run_id), purpose=kind)

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self.lock:
            kind = self.model_kinds.pop(str(run_id), "decision")
            counts = self.usage["model_breakdown"].get(kind)
            usages = [getattr(generation.message, "usage_metadata", None)
                      for batch in response.generations for generation in batch
                      if hasattr(generation, "message")]
            if not any(usages):
                self.usage["usage_unavailable_calls"] += 1
                if counts is not None:
                    counts["usage_unavailable_calls"] += 1
            for usage in filter(None, usages):
                self.usage["input_tokens"] += usage.get("input_tokens", 0)
                self.usage["output_tokens"] += usage.get("output_tokens", 0)
                if counts is not None:
                    counts["input_tokens"] += usage.get("input_tokens", 0)
                    counts["output_tokens"] += usage.get("output_tokens", 0)
            self.save()
        messages = [{"content": _visible_content(generation.message.content), "tool_calls": generation.message.tool_calls}
                    for batch in response.generations for generation in batch if hasattr(generation, "message")]
        self.event("model_completed", call_id=str(run_id), purpose=kind, usage=usages, messages=messages)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self.lock:
            kind = self.model_kinds.pop(str(run_id), "decision")
        self.event("model_failed", call_id=str(run_id), purpose=kind, error_type=type(error).__name__)


def _visible_content(content):
    if isinstance(content, str):
        return content
    return [block if isinstance(block, str) else {"type": "text", "text": block.get("text", "")}
            for block in content if isinstance(block, str) or (
                isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            )]


class DomainMiddleware(AgentMiddleware):
    state_schema = ExecutionState

    def __init__(self, ledger: RunLedger, *, requires_confirmation=False, skills=()):
        self.ledger = ledger
        self.requires_confirmation = requires_confirmation
        self.tool_lock = asyncio.Lock()
        self.skills = skills

    async def awrap_tool_call(self, request, handler):
        async with self.tool_lock:
            call = request.tool_call
            started = time.monotonic()
            metadata = (request.tool.metadata or {}) if request.tool is not None else {}
            service = metadata.get("service")
            error_category = None
            self.ledger.reserve("tool")
            self.ledger.event("tool_started", call_id=call["id"], name=call["name"], arguments=call["args"],
                              service=service, remote_name=metadata.get("remote_name"))
            legal = call["name"] in {"search_knowledge", "fetch_knowledge"} or (
                call["name"].startswith("pkulaw_") and not call["name"].startswith("pkulaw_account__")
            )
            confirmed = request.state.get("facts_confirmation", {}).get("confirmed", False)
            if legal and self.requires_confirmation and not confirmed:
                result = ToolMessage(content="Fact confirmation required. Use ask_user with facts_summary before legal retrieval.",
                                     tool_call_id=call["id"], name=call["name"], status="error")
            else:
                timeout = self.ledger.limits.tool_timeout_sec
                if service:
                    self.ledger.record_mcp(service, "started")
                if request.tool is not None and request.tool.metadata:
                    timeout = min(timeout, request.tool.metadata.get("timeout_sec", timeout))
                try:
                    result = await asyncio.wait_for(handler(request), timeout=timeout)
                except GraphInterrupt:
                    raise
                except (BudgetExceeded, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    error_category = failure_category(exc) if service else "tool_error"
                    result = ToolMessage(content=f"Tool failed: {type(exc).__name__} ({error_category})", tool_call_id=call["id"],
                                         name=call["name"], status="error")
                if service:
                    artifact = result.artifact if isinstance(result, ToolMessage) else None
                    structured = artifact.get("structured_content") if isinstance(artifact, dict) else None
                    if isinstance(result, ToolMessage) and (response_failed(result.content) or response_failed(structured)):
                        content = result.content
                        if response_failed(structured) and not response_failed(content):
                            notice = "MCP structured result reports business failure; do not use it as successful evidence."
                            content = content + "\n" + notice if isinstance(content, str) else [*content, {"type": "text", "text": notice}]
                        result = result.model_copy(update={"status": "error", "content": content})
                        error_category = "business"
                    elif getattr(result, "status", "success") == "error" and error_category is None:
                        error_category = "tool_error"
                    self.ledger.record_mcp(service, getattr(result, "status", "success"))
            content = getattr(result, "content", None)
            artifact = getattr(result, "artifact", None) if service else None
            response_hash = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                                      default=str).encode()).hexdigest() if service else None
            artifact_hash = hashlib.sha256(json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                                                      default=str).encode()).hexdigest() if artifact is not None else None
            self.ledger.event("tool_completed", call_id=call["id"], name=call["name"], service=service,
                              duration_sec=time.monotonic() - started, response_sha256=response_hash,
                              error_category=error_category,
                              artifact=artifact, artifact_sha256=artifact_hash,
                              status=getattr(result, "status", "success"), result=content)
            if call["name"] == "read_file" and getattr(result, "status", "success") == "success":
                path = call["args"].get("file_path", "")
                for skill in self.skills:
                    if path == skill["path"] or path.startswith(str(Path(skill["path"]).parent) + "/"):
                        self.ledger.event("skill_read", skill_id=skill["id"], path=path, call_id=call["id"],
                                          kind="instructions" if path == skill["path"] else "resource")
            if call["name"] in {"search_knowledge", "fetch_knowledge"} and not self.ledger.paths.compact:
                with (self.ledger.run / "citations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"call_id": call["id"], "tool": call["name"], "result": getattr(result, "content", None)},
                                            ensure_ascii=False, default=str) + "\n")
            return result
