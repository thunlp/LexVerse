from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from lexverse.capabilities.registry import ResourceRegistry
from lexverse.config import ConfigError, load_profile
from lexverse.models.trial import MCPServerConfig
from lexverse.runtime.paths import AGENT_RUNTIME_ROOT, PROJECT_ROOT, RunPaths
from lexverse.runtime.results import atomic_write_json


def failure_category(error: Exception) -> str:
    """Classify transport errors without exposing request details."""
    import httpx
    pending, seen, categories = [error], set(), set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            status = current.response.status_code
            categories.add("authorization" if status in {401, 403} else "service")
        elif isinstance(current, (TimeoutError, httpx.TimeoutException)):
            categories.add("timeout")
        elif isinstance(current, httpx.TransportError):
            categories.add("network")
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        cause = current.__cause__ or current.__context__
        if cause is not None:
            pending.append(cause)
    return next((category for category in ("authorization", "timeout", "network", "service")
                 if category in categories), "tool_error")


def response_failed(content) -> bool:
    """Recognize explicit failure flags in top-level response envelopes."""
    values = content if isinstance(content, list) else [content]
    for value in values:
        if isinstance(value, dict) and value.get("type") == "text":
            value = value.get("text", "")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict) and (
            value.get("isError") is True or value.get("success") is False
        ):
            return True
    return False


def tool_name(alias: str, remote_name: str) -> str:
    name = f"{alias}__{remote_name}"
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        return name
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:47]
    return readable + "_" + hashlib.sha256(name.encode()).hexdigest()[:16]


@asynccontextmanager
async def open_mcp_tools(registry: ResourceRegistry, selected: list[str], *, adapter_factory=None):
    if adapter_factory is None:
        from langchain.mcp import MCPAdapter
        adapter_factory = MCPAdapter
    if len(selected) != len(set(selected)):
        raise ConfigError("Duplicate MCP selections")
    tools, services, names = [], [], set()
    async with AsyncExitStack() as stack:
        for alias in selected:
            service = registry.mcp.get(alias)
            if service is None:
                raise ConfigError(f"Unknown MCP service: {alias}")
            config = {"url": service.url, "transport": service.transport}
            try:
                if service.credential_ref:
                    config["headers"] = {"Authorization": "Bearer " + load_profile(service.credential_ref).api_key}
                adapter = await asyncio.wait_for(
                    stack.enter_async_context(adapter_factory({"mcpServers": {alias: config}})),
                    timeout=service.timeout_sec,
                )
                discovered = await asyncio.wait_for(adapter.list_tools(), timeout=service.timeout_sec)
            except Exception as exc:
                category = failure_category(exc)
                if service.required:
                    raise ConfigError(f"MCP preparation failed: {alias} ({category}; {type(exc).__name__})") from None
                services.append({"alias": alias, "status": "unavailable", "error_type": type(exc).__name__,
                                 "error_category": category})
                continue
            remote_names = {tool.name for tool in discovered}
            if service.allowed_tools is not None and set(service.allowed_tools) - remote_names:
                raise ConfigError(f"MCP allowlist contains missing tools: {alias}")
            report = {"alias": alias, "url": service.url, "credential_ref": service.credential_ref,
                      "status": "available", "discovered": sorted(remote_names), "tools": []}
            for remote in discovered:
                if service.allowed_tools is not None and remote.name not in service.allowed_tools:
                    continue
                name = tool_name(alias, remote.name)
                if name in names:
                    raise ConfigError(f"Duplicate model tool name: {name}")
                names.add(name)
                schema = remote.args_schema if isinstance(remote.args_schema, dict) else remote.get_input_schema().model_json_schema()
                schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                identity = {"service": alias, "remote_name": remote.name, "schema_hash": schema_hash,
                            "timeout_sec": float(service.timeout_sec)}
                tools.append(remote.model_copy(update={"name": name, "metadata": identity}))
                report["tools"].append({"name": name, **identity, "description": remote.description, "schema": schema})
            services.append(report)
        yield tools, services


def build_knowledge_mcp_config(spec, manifest_path, resources, *, cache_dir=None, env_type="local"):
    """Describe the same knowledge service for built-in and external clients."""
    selected = set(spec.selection.tools) & {"search_knowledge", "fetch_knowledge"}
    if not spec.selection.knowledge or not selected:
        return None
    public_manifest = manifest_path.parent / "knowledge_indexes.json"
    atomic_write_json(public_manifest, {"knowledge": resources.get("knowledge", []),
                                       "keyword_cache": resources.get("keyword_cache"),
                                       "tools": sorted(selected)})
    args = ["-m", "lexverse.capabilities.knowledge.mcp_server", "--manifest", str(public_manifest.resolve())]
    if cache_dir is not None:
        args.extend(["--cache", str(cache_dir.resolve())])
    pythonpath = ("/lexverse-repo" if env_type == "docker" else
                  os.pathsep.join([str(AGENT_RUNTIME_ROOT / "site-packages"), str(PROJECT_ROOT)]))
    return MCPServerConfig(name="lexverse_knowledge", transport="stdio",
                           command="python3" if env_type == "docker" else sys.executable, args=args,
                           env={"PYTHONPATH": pythonpath})


@asynccontextmanager
async def open_knowledge_tools(spec, resources, *, cache, embedding_recorder=None):
    """Keep the local knowledge MCP process alive only for this execution."""
    from fastmcp.client.transports import StdioTransport
    from langchain.mcp import MCPAdapter
    from langchain_core.tools import StructuredTool, ToolException

    paths = RunPaths(spec.run_dir)
    server = build_knowledge_mcp_config(spec, paths.internal / "manifest.json", resources, cache_dir=cache)
    if server is None:
        yield []
        return
    # Child interpreters need the same bundled dependencies as the parent.
    pythonpath = os.pathsep.join(str(Path(path).resolve()) for path in sys.path if path)
    with TemporaryDirectory(prefix="knowledge-mcp-", dir=paths.internal) as temporary:
        usage_path = Path(temporary) / "embedding.jsonl"
        usage_path.touch()
        position = 0

        def collect_usage():
            nonlocal position
            with usage_path.open(encoding="utf-8") as stream:
                stream.seek(position)
                for line in stream:
                    if embedding_recorder is not None:
                        embedding_recorder(**json.loads(line))
                position = stream.tell()

        transport = StdioTransport(server.command, [*server.args, "--usage-log", str(usage_path)],
                                   env={**os.environ, "PYTHONPATH": pythonpath}, keep_alive=False)
        try:
            async with MCPAdapter(transport) as adapter:
                discovered = {tool.name: tool for tool in await adapter.list_tools()}
                selected = set(spec.selection.tools) & {"search_knowledge", "fetch_knowledge"}
                if selected - discovered.keys():
                    raise ConfigError("Knowledge MCP service is missing selected tools")

                def wrap(remote):
                    async def invoke(**arguments):
                        try:
                            response = await remote.ainvoke({"name": remote.name, "args": arguments,
                                                             "id": "knowledge", "type": "tool_call"})
                            if response.status == "error":
                                raise ToolException(str(response.content))
                            return json.loads(response.content[0]["text"])
                        finally:
                            collect_usage()

                    return StructuredTool(name=remote.name, description=remote.description,
                                          args_schema=remote.args_schema, coroutine=invoke,
                                          metadata={"service": "lexverse_knowledge", "remote_name": remote.name})

                yield [wrap(discovered[name]) for name in sorted(selected)]
        finally:
            collect_usage()
