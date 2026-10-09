from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
import hashlib
import json
import re

from lexverse.capabilities.registry import ResourceRegistry
from lexverse.config import ConfigError, load_profile


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
    if "pkulaw_law_agg" in selected and set(selected) & {
        "pkulaw_law_search", "pkulaw_law_keyword", "pkulaw_law_item_keyword",
    }:
        raise ConfigError("Choose aggregated or individual law services, not both")
    tools, services, names = [], [], set()
    async with AsyncExitStack() as stack:
        for alias in selected:
            service = registry.mcp.get(alias)
            if service is None:
                raise ConfigError(f"Unknown MCP service: {alias}")
            config = {"url": service.url}
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
