"""Provider adapters for the read-only LexVerse environment."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .errors import (
    InvalidArgumentError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ReadLimitExceededError,
    UnknownProviderError,
)
from .query import SearchRequest
from .utils import short_value


class Provider(Protocol):
    name: str

    def search(self, request: SearchRequest) -> dict[str, Any]:
        ...

    def get(self, record_id: str) -> dict[str, Any]:
        ...


@dataclass
class LocalProvider:
    """Provider backed by the local read-only index and record store."""

    query_service: Any
    record_store: Any
    name: str = "local"

    @property
    def available(self) -> bool:
        return True

    def search(self, request: SearchRequest) -> dict[str, Any]:
        return self.query_service.search(request)

    def get(self, record_id: str) -> dict[str, Any]:
        return self.record_store.get_record(record_id)

    def read_record_part(self, record_id: str, path: str, *, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        return self.record_store.read_record_part(record_id, path, offset=offset, limit=limit)

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        return self.record_store.get_asset(asset_id, mode=mode)


def _safe_remote_key_fields(value: Any) -> dict[str, Any]:
    """Keep search responses small and free of remote snippets/scores."""

    if not isinstance(value, dict):
        return {}
    allowed = {
        "title", "law_name", "article", "case_number", "case_cause", "court",
        "stage", "category", "effective_from", "source", "year",
        "template_type", "case_type", "term", "question", "theme",
    }
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in allowed:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = short_value(item)
        elif isinstance(item, list) and all(isinstance(child, (str, int, float, bool)) or child is None for child in item):
            result[key] = [short_value(child) for child in item[:20]]
    return result


class PkulawMcpProvider:
    """Optional adapter for a caller-supplied 北大法宝 MCP bridge."""

    name = "pkulaw"

    def __init__(self, call_tool: Callable[..., Any] | None = None, *,
                 service_id: str | None = None,
                 search_tool: str | None = None,
                 get_tool: str | None = None,
                 map_search: Callable[[SearchRequest], dict[str, Any]] | None = None,
                 map_get: Callable[[str], dict[str, Any]] | None = None,
                 timeout_seconds: float = 30.0,
                 max_response_bytes: int = 12 * 1024 * 1024):
        self.call_tool = call_tool
        self.service_id = service_id or "default"
        self.search_tool = search_tool
        self.get_tool = get_tool
        self.map_search = map_search
        self.map_get = map_get
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise InvalidArgumentError("timeout_seconds 必须大于 0")
        self.timeout_seconds = timeout_seconds
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise InvalidArgumentError("max_response_bytes 必须是正整数")
        self.max_response_bytes = max_response_bytes

    @property
    def id_prefix(self) -> str:
        return f"lv:external:pkulaw:{self.service_id}:"

    @property
    def available(self) -> bool:
        return (
            self.call_tool is not None
            and bool(self.search_tool)
            and bool(self.get_tool)
        )

    def _ensure_available(self) -> Callable[..., Any]:
        if self.call_tool is None:
            raise ProviderUnavailableError("北大法宝 Provider 未配置 MCP 连接", details={"provider": self.name})
        return self.call_tool

    def _call(self, tool_name: str | None, arguments: dict[str, Any]) -> Any:
        if not tool_name:
            raise ProviderUnavailableError(
                "北大法宝 Provider 未配置已验证的工具名",
                details={"provider": self.name},
            )
        call_tool = self._ensure_available()
        try:
            try:
                parameters = inspect.signature(call_tool).parameters.values()
                accepts_timeout = any(
                    parameter.name == "timeout_seconds"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                accepts_timeout = False
            if accepts_timeout:
                result = call_tool(
                    tool_name, arguments, timeout_seconds=self.timeout_seconds
                )
            else:
                result = call_tool(tool_name, arguments)
            if inspect.isawaitable(result):
                raise ProviderUnavailableError("MCP 回调返回异步结果；请注入同步桥接函数", details={"provider": self.name})
            try:
                response_size = len(
                    json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
                )
            except (TypeError, ValueError) as exc:
                raise ProviderUnavailableError(
                    "北大法宝 MCP 返回了无法序列化的结果",
                    details={"provider": self.name, "tool": tool_name},
                ) from exc
            if response_size > self.max_response_bytes:
                raise ReadLimitExceededError(
                    "北大法宝 MCP 返回结果超过大小限制",
                    details={"max_response_bytes": self.max_response_bytes},
                )
            # Accept a raw JSON-RPC MCP response as well as a bridge that
            # already returns the tool's structured payload.
            if isinstance(result, dict) and "jsonrpc" in result:
                if isinstance(result.get("error"), dict):
                    raise ProviderUnavailableError(
                        "北大法宝 MCP 返回 JSON-RPC 错误",
                        details={"provider": self.name, "tool": tool_name},
                    )
                rpc_result = result.get("result")
                if isinstance(rpc_result, dict) and rpc_result.get("isError"):
                    raise ProviderUnavailableError(
                        "北大法宝 MCP 工具调用失败",
                        details={"provider": self.name, "tool": tool_name},
                    )
                if isinstance(rpc_result, dict) and "structuredContent" in rpc_result:
                    result = rpc_result["structuredContent"]
                elif isinstance(rpc_result, dict):
                    result = rpc_result
            return result
        except (ProviderTimeoutError, ProviderUnavailableError, ReadLimitExceededError):
            raise
        except TimeoutError as exc:
            raise ProviderTimeoutError("北大法宝 MCP 调用超时", details={"provider": self.name, "tool": tool_name}) from exc
        except Exception as exc:
            raise ProviderUnavailableError("北大法宝 MCP 调用失败", details={"provider": self.name, "tool": tool_name}) from exc

    def _remote_id(self, record_id: str) -> str:
        if not isinstance(record_id, str) or not record_id.startswith(self.id_prefix):
            raise InvalidArgumentError("不是当前北大法宝 Provider 的记录 id")
        remote_id = record_id[len(self.id_prefix):]
        if not remote_id:
            raise InvalidArgumentError("外部记录 id 为空")
        return remote_id

    def _public_id(self, remote_id: Any) -> str:
        if remote_id in (None, ""):
            raise ProviderUnavailableError("北大法宝结果缺少记录 id", details={"provider": self.name})
        return f"{self.id_prefix}{remote_id}"

    @staticmethod
    def _items(payload: Any) -> tuple[list[Any], Any]:
        if isinstance(payload, dict):
            items = payload.get(
                "items",
                payload.get("results", payload.get("data", payload.get("result", []))),
            )
            cursor = payload.get("next_cursor", payload.get("nextCursor"))
            if isinstance(items, dict):
                items = items.get("items", items.get("results", items.get("data", [])))
            return (items if isinstance(items, list) else [], cursor)
        return (payload if isinstance(payload, list) else [], None)

    def search(self, request: SearchRequest) -> dict[str, Any]:
        arguments = self.map_search(request) if self.map_search else {
            "collection": request.collection,
            "query": request.query,
            "filters": request.filters or {},
            "sort": request.sort,
            "limit": request.limit,
            **({"cursor": request.cursor} if request.cursor else {}),
        }
        payload = self._call(self.search_tool, arguments)
        raw_items, next_cursor = self._items(payload)
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise ProviderUnavailableError(
                "北大法宝搜索返回了无效的 cursor",
                details={"provider": self.name},
            )
        items: list[dict[str, Any]] = []
        for raw in raw_items[: request.limit]:
            if isinstance(raw, dict):
                remote_id = raw.get(
                    "id", raw.get("record_id", raw.get("key", raw.get("gid")))
                )
                fields = raw.get("key_fields", raw.get("metadata", raw))
            else:
                remote_id, fields = raw, {}
            item: dict[str, Any] = {"id": self._public_id(remote_id)}
            key_fields = _safe_remote_key_fields(fields)
            if key_fields:
                item["key_fields"] = key_fields
            items.append(item)
        return {"items": items, "next_cursor": next_cursor}

    def get(self, record_id: str) -> dict[str, Any]:
        remote_id = self._remote_id(record_id)
        arguments = self.map_get(remote_id) if self.map_get else {"id": remote_id}
        payload = self._call(self.get_tool, arguments)
        if isinstance(payload, dict) and "record" in payload:
            envelope: dict[str, Any] = {"id": record_id, "record": payload["record"]}
            if isinstance(payload.get("assets"), list):
                envelope["assets"] = payload["assets"]
        else:
            envelope = {"id": record_id, "record": payload}
        envelope["provenance"] = {"provider": self.name, "service_id": self.service_id, "remote_id": remote_id}
        return envelope

    def read_record_part(self, record_id: str, path: str, *, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        raise InvalidArgumentError("北大法宝 Provider 尚未声明 read_record_part；请先 get_record")

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        raise InvalidArgumentError("北大法宝 Provider 不提供本地资产读取")


class ProviderRouter:
    """Route explicit provider calls and provider-prefixed record ids."""

    def __init__(self, providers: list[Provider] | tuple[Provider, ...]):
        self._providers = {provider.name: provider for provider in providers}

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def available_names(self) -> tuple[str, ...]:
        return tuple(sorted(
            name for name, provider in self._providers.items()
            if getattr(provider, "available", True)
        ))

    def get(self, name: str) -> Provider:
        if not isinstance(name, str) or not name:
            raise InvalidArgumentError("provider 必须是非空字符串")
        provider = self._providers.get(name)
        if provider is None:
            raise UnknownProviderError(f"未知 Provider: {name}")
        return provider

    def for_record(self, record_id: str) -> Provider:
        if not isinstance(record_id, str) or not record_id:
            raise InvalidArgumentError("record_id 必须是非空字符串")
        if isinstance(record_id, str) and record_id.startswith("lv:external:pkulaw:"):
            return self.get("pkulaw")
        return self.get("local")

    def search(self, request: SearchRequest) -> dict[str, Any]:
        return self.get(request.provider).search(request)

    def get_record(self, record_id: str) -> dict[str, Any]:
        return self.for_record(record_id).get(record_id)

    def read_record_part(self, record_id: str, path: str, *, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        method = getattr(self.for_record(record_id), "read_record_part", None)
        if method is None:
            raise InvalidArgumentError("当前 Provider 不支持 read_record_part")
        return method(record_id, path, offset=offset, limit=limit)

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        return self.get("local").get_asset(asset_id, mode=mode)
