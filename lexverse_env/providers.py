"""Provider adapters for the read-only LexVerse environment."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Protocol
from urllib.request import Request, urlopen

from .errors import (
    InvalidArgumentError,
    InvalidCursorError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RecordNotFoundError,
    ReadLimitExceededError,
    UnknownProviderError,
)
from .query import SearchRequest
from .utils import decode_cursor, encode_cursor, short_value, stable_hash


PKULAW_SERVICE_ENDPOINTS = {
    "law_semantic": "https://apim-gateway.pkulaw.com/mcp-law-search-service",
    "law_keyword": "https://apim-gateway.pkulaw.com/mcp-law",
    "case_semantic": "https://apim-gateway.pkulaw.com/mcp-case-search-service",
    "case_keyword": "https://apim-gateway.pkulaw.com/mcp-case",
}
PKULAW_SERVICE_ENV_VARS = {
    name: f"PKULAW_{name.upper()}_MCP_ENDPOINT"
    for name in PKULAW_SERVICE_ENDPOINTS
}
DEFAULT_PKULAW_ENDPOINT = PKULAW_SERVICE_ENDPOINTS["law_semantic"]
DEFAULT_PKULAW_SERVICE_ID = "mcp-law-search-service"
DEFAULT_PKULAW_SEARCH_TOOL = "search_article"
DEFAULT_PKULAW_GET_TOOL = "get_article"
_PKULAW_COLLECTIONS = {"legal_laws", "legal_cases"}
_REMOTE_URL_ID = re.compile(r"/(?:chl|lar|pfnl)/([^/?#]+)\.html", re.I)


def parse_pkulaw_mcp_response(raw: str) -> dict[str, Any]:
    """Parse a JSON or server-sent-events response from the Pkulaw gateway."""

    for line in raw.splitlines():
        if line.startswith("data:"):
            message = json.loads(line[5:].lstrip())
            if isinstance(message, dict) and (
                "result" in message or "error" in message
            ):
                return message
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("北大法宝 MCP 返回的不是 JSON 对象")
    return result


def call_pkulaw_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float = 30,
    *,
    token: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Synchronously call a tool on the Pkulaw HTTP MCP gateway."""

    authorization = token or os.environ.get("PKULAW_TOKEN")
    if not authorization:
        raise ProviderUnavailableError(
            "缺少 PKULAW_TOKEN；请配置北大法宝访问令牌",
            details={"provider": "pkulaw"},
        )
    gateway = endpoint or os.environ.get(
        "PKULAW_LAW_MCP_ENDPOINT", DEFAULT_PKULAW_ENDPOINT
    )
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    request = Request(
        gateway,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        result = parse_pkulaw_mcp_response(response.read().decode("utf-8"))
    if result.get("error"):
        raise ProviderUnavailableError(
            "北大法宝 MCP 调用失败",
            details={"provider": "pkulaw", "tool": tool_name},
        )
    return result


def _map_pkulaw_search(request: SearchRequest) -> dict[str, Any]:
    return {"text": request.query or "", "size": request.limit}


def _map_pkulaw_get(remote_id: str) -> dict[str, Any]:
    return {"gid": remote_id}


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


class LayeredLocalProvider:
    """Merge isolated user and official local indexes behind one provider."""

    name = "local"

    def __init__(
        self,
        layers: tuple[tuple[str, LocalProvider], ...],
        *,
        max_limit: int = 50,
    ):
        if not layers:
            raise InvalidArgumentError("本地索引层不能为空")
        self.layers = layers
        self.max_limit = max_limit
        self.generation = stable_hash(
            [
                [name, layer.query_service.meta.get("index_generation")]
                for name, layer in layers
            ],
            length=32,
        )

    @property
    def available(self) -> bool:
        return True

    def _fetch(self, layer: LocalProvider, request: SearchRequest, target: int) -> tuple[list[dict[str, Any]], bool]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        while len(items) < target:
            page = layer.search(SearchRequest(
                collection=request.collection,
                provider="local",
                query=request.query,
                filters=request.filters,
                sort=request.sort,
                limit=min(self.max_limit, max(1, target - len(items))),
                cursor=cursor,
            ))
            items.extend(page.get("items", []))
            cursor = page.get("next_cursor")
            if not cursor:
                return items, False
        return items, cursor is not None

    @staticmethod
    def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            record_id = item.get("id")
            if not isinstance(record_id, str) or record_id in seen:
                continue
            seen.add(record_id)
            result.append(item)
        return result

    @staticmethod
    def _interleave(pages: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for offset in range(max((len(page) for page in pages), default=0)):
            for page in pages:
                if offset < len(page):
                    result.append(page[offset])
        return result

    def search(self, request: SearchRequest) -> dict[str, Any]:
        request_hash = stable_hash(request.canonical(), length=32)
        offset = 0
        if request.cursor:
            payload = decode_cursor(request.cursor)
            if (
                payload.get("version") != 1
                or payload.get("request_hash") != request_hash
                or payload.get("index_generation") != self.generation
            ):
                raise InvalidCursorError("cursor 与当前分层查询或索引版本不匹配")
            offset = payload.get("offset", -1)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise InvalidCursorError("cursor offset 无效")
        target = offset + request.limit + 1
        fetched = [
            self._fetch(layer, request, target) for _, layer in self.layers
        ]
        pages = [page for page, _ in fetched]
        if request.sort == "stable" or not request.query:
            merged = self._deduplicate([item for page in pages for item in page])
            merged.sort(key=lambda item: item["id"])
        else:
            merged = self._deduplicate(self._interleave(pages))
        page = merged[offset : offset + request.limit]
        has_more = len(merged) > offset + len(page) or any(more for _, more in fetched)
        next_cursor = None
        if has_more:
            next_cursor = encode_cursor({
                "version": 1,
                "request_hash": request_hash,
                "index_generation": self.generation,
                "offset": offset + len(page),
            })
        return {"items": page, "next_cursor": next_cursor}

    def _record_layer(self, record_id: str) -> tuple[str, LocalProvider]:
        for name, layer in self.layers:
            try:
                layer.record_store._record_row(record_id)
                return name, layer
            except RecordNotFoundError:
                continue
        raise RecordNotFoundError(f"记录不存在: {record_id}")

    def get(self, record_id: str) -> dict[str, Any]:
        name, layer = self._record_layer(record_id)
        result = layer.get(record_id)
        result.setdefault("provenance", {})["index_layer"] = name
        return result

    def read_record_part(self, record_id: str, path: str, *, offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        name, layer = self._record_layer(record_id)
        result = layer.read_record_part(
            record_id, path, offset=offset, limit=limit
        )
        result["index_layer"] = name
        return result

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        for name, layer in self.layers:
            try:
                result = layer.get_asset(asset_id, mode=mode)
                result["index_layer"] = name
                return result
            except RecordNotFoundError:
                continue
        raise RecordNotFoundError(f"资产不存在: {asset_id}")


def _safe_remote_key_fields(value: Any) -> dict[str, Any]:
    """Keep search responses small and free of remote snippets/scores."""

    if not isinstance(value, dict):
        return {}
    allowed = {
        "title", "law_name", "article", "case_number", "case_cause", "court",
        "stage", "category", "effective_from", "source", "year",
        "template_type", "case_type", "term", "question", "theme", "doc_type",
        "case_grade", "decision_date", "cause_of_action", "timeliness",
        "effectiveness", "issue_department", "issue_date", "implementation_date",
        "doc_no", "url",
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
    """Adapter for Pkulaw's collection search and legal-text MCP services."""

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
        self._service_endpoints: dict[str, str] = {}
        self._authorization: str | None = None
        self._record_cache: dict[str, Any] = {}
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

    @classmethod
    def standard(
        cls,
        *,
        call_tool: Callable[..., Any] | None = None,
        token: str | None = None,
        endpoint: str | None = None,
        endpoints: dict[str, str] | None = None,
        service_id: str | None = None,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 12 * 1024 * 1024,
    ) -> "PkulawMcpProvider":
        """Build the standard multi-service Pkulaw adapter.

        A supplied bridge receives the documented tool names. Without one, the
        provider sends HTTP MCP calls to the service endpoint selected for the
        requested collection or legal-text operation.
        """

        authorization = token or os.environ.get("PKULAW_TOKEN")
        configured_endpoints = dict(PKULAW_SERVICE_ENDPOINTS)
        for name, variable in PKULAW_SERVICE_ENV_VARS.items():
            configured_endpoints[name] = os.environ.get(
                variable, configured_endpoints[name]
            )
        # Preserve the endpoint variable and argument introduced for the
        # original law-semantic-only adapter.
        configured_endpoints["law_semantic"] = (
            endpoint
            or os.environ.get("PKULAW_LAW_MCP_ENDPOINT")
            or configured_endpoints["law_semantic"]
        )
        if endpoints is not None and not isinstance(endpoints, dict):
            raise InvalidArgumentError("pkulaw_endpoints 必须是对象")
        if endpoints:
            unknown = set(endpoints) - set(PKULAW_SERVICE_ENDPOINTS)
            if unknown:
                raise InvalidArgumentError(
                    f"未知北大法宝服务 endpoint: {sorted(unknown)[0]}"
                )
            for name, value in endpoints.items():
                if not isinstance(value, str) or not value:
                    raise InvalidArgumentError(
                        f"北大法宝服务 endpoint 必须是非空字符串: {name}"
                    )
            configured_endpoints.update(endpoints)
        bridge = call_tool
        if bridge is None and authorization:
            bridge = partial(
                call_pkulaw_mcp_tool,
                token=authorization,
                endpoint=configured_endpoints["law_semantic"],
            )
        provider = cls(
            bridge,
            service_id=service_id or DEFAULT_PKULAW_SERVICE_ID,
            search_tool=DEFAULT_PKULAW_SEARCH_TOOL,
            get_tool=DEFAULT_PKULAW_GET_TOOL,
            map_search=_map_pkulaw_search,
            map_get=_map_pkulaw_get,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        provider._service_endpoints = configured_endpoints
        provider._authorization = authorization if call_tool is None else None
        return provider

    @classmethod
    def from_env(
        cls,
        *,
        token: str | None = None,
        endpoint: str | None = None,
        endpoints: dict[str, str] | None = None,
        service_id: str | None = None,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 12 * 1024 * 1024,
    ) -> "PkulawMcpProvider":
        """Build the standard adapter from arguments and environment."""

        return cls.standard(
            token=token,
            endpoint=endpoint,
            endpoints=endpoints,
            service_id=service_id,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )

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

    def _call(
        self,
        tool_name: str | None,
        arguments: dict[str, Any],
        *,
        bridge: Callable[..., Any] | None = None,
    ) -> Any:
        if not tool_name:
            raise ProviderUnavailableError(
                "北大法宝 Provider 未配置已验证的工具名",
                details={"provider": self.name},
            )
        call_tool = bridge or self._ensure_available()
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
                elif isinstance(rpc_result, dict) and isinstance(
                    rpc_result.get("content"), list
                ):
                    text_parts = [
                        item.get("text")
                        for item in rpc_result["content"]
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                        and isinstance(item.get("text"), str)
                    ]
                    if len(text_parts) == 1:
                        try:
                            result = json.loads(text_parts[0])
                        except json.JSONDecodeError:
                            result = text_parts[0]
                    else:
                        result = rpc_result
                elif isinstance(rpc_result, dict):
                    result = rpc_result
            return result
        except (ProviderTimeoutError, ProviderUnavailableError, ReadLimitExceededError):
            raise
        except TimeoutError as exc:
            raise ProviderTimeoutError("北大法宝 MCP 调用超时", details={"provider": self.name, "tool": tool_name}) from exc
        except Exception as exc:
            raise ProviderUnavailableError("北大法宝 MCP 调用失败", details={"provider": self.name, "tool": tool_name}) from exc

    def _call_service(
        self, service: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        if service not in PKULAW_SERVICE_ENDPOINTS:
            raise InvalidArgumentError(f"未知北大法宝服务: {service}")
        if self._authorization:
            endpoint = self._service_endpoints[service]
            bridge = partial(
                call_pkulaw_mcp_tool,
                token=self._authorization,
                endpoint=endpoint,
            )
            return self._call(tool_name, arguments, bridge=bridge)
        return self._call(tool_name, arguments)

    def _remote_id(self, record_id: str) -> str:
        if not isinstance(record_id, str):
            raise InvalidArgumentError("不是当前北大法宝 Provider 的记录 id")
        if record_id.startswith(self.id_prefix):
            remote_id = record_id[len(self.id_prefix):]
        elif self._service_endpoints and record_id.startswith("lv:external:pkulaw:"):
            parts = record_id.split(":", 4)
            if len(parts) != 5 or parts[3] not in self._service_endpoints:
                raise InvalidArgumentError("不是当前北大法宝 Provider 的记录 id")
            remote_id = parts[4]
        else:
            raise InvalidArgumentError("不是当前北大法宝 Provider 的记录 id")
        if not remote_id:
            raise InvalidArgumentError("外部记录 id 为空")
        return remote_id

    def _public_id(self, remote_id: Any, *, namespace: str | None = None) -> str:
        if remote_id in (None, ""):
            raise ProviderUnavailableError("北大法宝结果缺少记录 id", details={"provider": self.name})
        prefix = (
            f"lv:external:pkulaw:{namespace}:" if namespace else self.id_prefix
        )
        return f"{prefix}{remote_id}"

    @staticmethod
    def _items(payload: Any) -> tuple[list[Any], Any]:
        if isinstance(payload, dict):
            items = payload.get(
                "items",
                payload.get(
                    "results",
                    payload.get(
                        "data",
                        payload.get("Data", payload.get("result", [])),
                    ),
                ),
            )
            cursor = payload.get("next_cursor", payload.get("nextCursor"))
            if isinstance(items, dict):
                items = items.get(
                    "items",
                    items.get(
                        "results", items.get("data", items.get("Data", []))
                    ),
                )
            return (items if isinstance(items, list) else [], cursor)
        return (payload if isinstance(payload, list) else [], None)

    @staticmethod
    def _normalized_fields(raw: dict[str, Any]) -> dict[str, Any]:
        aliases = {
            "Title": "title",
            "Gid": "gid",
            "CaseFlag": "case_number",
            "Court": "court",
            "Category": "category",
            "CaseGrade": "case_grade",
            "DocumentAttr": "doc_type",
            "CaseClassName": "case_type",
            "LastInstanceDate": "decision_date",
            "IssueDate": "issue_date",
            "ImplementDate": "implementation_date",
            "TimelinessDic": "timeliness",
            "EffectivenessDic": "effectiveness",
            "DocumentNO": "doc_no",
            "FullText": "article",
            "Url": "url",
        }
        return {aliases.get(key, key): value for key, value in raw.items()}

    @staticmethod
    def _remote_key(raw: dict[str, Any], fields: dict[str, Any]) -> Any:
        for key in ("id", "record_id", "key", "gid", "Gid"):
            if raw.get(key) not in (None, ""):
                return raw[key]
        for key in ("case_number", "CaseFlag"):
            if fields.get(key) not in (None, ""):
                return fields[key]
        url = fields.get("url") or fields.get("Url")
        if isinstance(url, str):
            match = _REMOTE_URL_ID.search(url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _validated_filters(
        request: SearchRequest, allowed: set[str]
    ) -> dict[str, Any]:
        filters = dict(request.filters or {})
        unknown = set(filters) - allowed - {"search_mode"}
        if unknown:
            raise InvalidArgumentError(
                f"北大法宝不支持过滤字段: {sorted(unknown)[0]}"
            )
        return filters

    def supports_collection(self, collection: str) -> bool:
        return collection in _PKULAW_COLLECTIONS

    def describe_collection(self, collection: str) -> dict[str, Any]:
        if collection == "legal_laws":
            return {
                "filter_fields": [
                    "search_mode", "lib", "timeliness", "issue_department",
                    "implement_date_start", "implement_date_end", "title",
                    "fulltext", "effectiveness",
                ],
                "search_fields": ["title", "article"],
                "capabilities": ["search", "get"],
            }
        if collection == "legal_cases":
            return {
                "filter_fields": [
                    "search_mode", "case_type", "doc_type", "courthouse_name",
                    "courthouse_province", "decision_date_start",
                    "decision_date_end", "title", "fulltext", "case_grade",
                    "court",
                ],
                "search_fields": ["title", "case_number", "facts"],
                "capabilities": ["search", "get"],
            }
        raise InvalidArgumentError(f"北大法宝不支持集合: {collection}")

    def _standard_search(self, request: SearchRequest) -> tuple[str, str, dict[str, Any]]:
        if request.collection == "legal_laws":
            allowed = {
                "lib", "timeliness", "issue_department", "implement_date_start",
                "implement_date_end", "title", "fulltext", "effectiveness",
            }
            filters = self._validated_filters(request, allowed)
            mode = filters.pop("search_mode", "semantic")
            if mode == "semantic":
                incompatible = set(filters) - {
                    "lib", "timeliness", "issue_department",
                    "implement_date_start", "implement_date_end",
                }
                if incompatible:
                    raise InvalidArgumentError(
                        f"法规语义检索不支持过滤字段: {sorted(incompatible)[0]}"
                    )
                arguments = {"text": request.query or "", "size": request.limit}
                arguments.update({
                    key: filters[key]
                    for key in (
                        "lib", "timeliness", "issue_department",
                        "implement_date_start", "implement_date_end",
                    )
                    if key in filters
                })
                return "law_semantic", "search_article", arguments
            if mode == "keyword":
                incompatible = set(filters) - {
                    "title", "fulltext", "implement_date_start",
                    "implement_date_end", "timeliness", "effectiveness",
                }
                if incompatible:
                    raise InvalidArgumentError(
                        f"法规关键词检索不支持过滤字段: {sorted(incompatible)[0]}"
                    )
                arguments: dict[str, Any] = {}
                aliases = {
                    "title": "title", "fulltext": "fulltext",
                    "implement_date_start": "startImplementDate",
                    "implement_date_end": "endImplementDate",
                    "timeliness": "timeliness", "effectiveness": "effectiveness",
                }
                for key, remote_key in aliases.items():
                    if key in filters:
                        arguments[remote_key] = filters[key]
                if request.query and "title" not in arguments and "fulltext" not in arguments:
                    arguments["title"] = request.query
                if "title" not in arguments and "fulltext" not in arguments:
                    raise InvalidArgumentError(
                        "北大法宝关键词法规检索需要 query、title 或 fulltext"
                    )
                return "law_keyword", "get_law_list", arguments
        elif request.collection == "legal_cases":
            allowed = {
                "case_type", "doc_type", "courthouse_name",
                "courthouse_province", "decision_date_start", "decision_date_end",
                "title", "fulltext", "case_grade", "court",
            }
            filters = self._validated_filters(request, allowed)
            mode = filters.pop("search_mode", "semantic")
            if mode == "semantic":
                incompatible = set(filters) - {
                    "case_type", "doc_type", "courthouse_name",
                    "courthouse_province", "decision_date_start",
                    "decision_date_end",
                }
                if incompatible:
                    raise InvalidArgumentError(
                        f"案例语义检索不支持过滤字段: {sorted(incompatible)[0]}"
                    )
                arguments = {"text": request.query or "", "size": request.limit}
                arguments.update({
                    key: filters[key]
                    for key in (
                        "case_type", "doc_type", "courthouse_name",
                        "courthouse_province", "decision_date_start",
                        "decision_date_end",
                    )
                    if key in filters
                })
                return "case_semantic", "search_case", arguments
            if mode == "keyword":
                incompatible = set(filters) - {
                    "title", "fulltext", "case_grade", "case_type",
                    "doc_type", "court", "decision_date_start",
                    "decision_date_end",
                }
                if incompatible:
                    raise InvalidArgumentError(
                        f"案例关键词检索不支持过滤字段: {sorted(incompatible)[0]}"
                    )
                arguments: dict[str, Any] = {}
                aliases = {
                    "title": "title", "fulltext": "fulltext",
                    "case_grade": "caseGrade", "case_type": "caseClassName",
                    "doc_type": "documentAttr", "court": "court",
                    "decision_date_start": "startLastInstanceDate",
                    "decision_date_end": "endLastInstanceDate",
                }
                for key, remote_key in aliases.items():
                    if key in filters:
                        arguments[remote_key] = filters[key]
                if request.query and "title" not in arguments and "fulltext" not in arguments:
                    arguments["title"] = request.query
                if "title" not in arguments and "fulltext" not in arguments:
                    raise InvalidArgumentError(
                        "北大法宝关键词案例检索需要 query、title 或 fulltext"
                    )
                return "case_keyword", "get_case_list", arguments
        else:
            raise InvalidArgumentError(f"北大法宝不支持集合: {request.collection}")
        raise InvalidArgumentError("search_mode 只能是 semantic 或 keyword")

    def search(self, request: SearchRequest) -> dict[str, Any]:
        selected_service: str | None = None
        if self._service_endpoints:
            service, tool, arguments = self._standard_search(request)
            selected_service = service
            payload = self._call_service(service, tool, arguments)
        else:
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
                raw_fields = raw.get("key_fields", raw.get("metadata", raw))
                fields = self._normalized_fields(
                    raw_fields if isinstance(raw_fields, dict) else raw
                )
                remote_id = self._remote_key(raw, fields)
                if (
                    selected_service == "law_semantic"
                    and remote_id not in (None, "")
                    and isinstance(fields.get("article"), str)
                ):
                    article_key = hashlib.sha256(
                        fields["article"].encode("utf-8")
                    ).hexdigest()[:12]
                    remote_id = f"{remote_id}~{article_key}"
            else:
                remote_id, fields = raw, {}
            item: dict[str, Any] = {
                "id": self._public_id(remote_id, namespace=selected_service)
            }
            key_fields = _safe_remote_key_fields(fields)
            if key_fields:
                item["key_fields"] = key_fields
            items.append(item)
            if isinstance(raw, dict) and self._service_endpoints:
                self._record_cache[item["id"]] = (selected_service, raw)
        return {"items": items, "next_cursor": next_cursor}

    def get(self, record_id: str) -> dict[str, Any]:
        remote_id = self._remote_id(record_id)
        if record_id in self._record_cache:
            service, cached_record = self._record_cache[record_id]
            return {
                "id": record_id,
                "record": cached_record,
                "provenance": {
                    "provider": self.name,
                    "service_id": service,
                    "remote_id": remote_id,
                },
            }
        if self._service_endpoints:
            raise ProviderUnavailableError(
                "北大法宝记录不在当前环境缓存中；请先重新搜索该记录",
                details={"provider": self.name},
            )
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
