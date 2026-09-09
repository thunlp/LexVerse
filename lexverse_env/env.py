"""Public Python entry point for the read-only LexVerse data environment."""

from __future__ import annotations

import os
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .errors import EnvironmentNotReadyError, InvalidArgumentError, UnknownCollectionError
from .index import (
    IndexBuilder,
    LOADER_VERSION,
    SCHEMA_VERSION,
    TOKENIZER_VERSION,
    compute_snapshot_id,
    load_manifest,
    open_readonly_database,
)
from .providers import LocalProvider, PkulawMcpProvider, ProviderRouter
from .query import FILTER_FIELDS_BY_COLLECTION, QueryService, SearchRequest
from .registry import SourceRegistry, public_collection_label
from .store import RecordStore
from .utils import validate_limit


class CorpusEnv:
    """Read-only facade shared by SDK, CLI and MCP."""

    def __init__(self, *, data_dir: str | Path, state_dir: str | Path,
                 db: sqlite3.Connection, manifest: dict[str, Any],
                 registry: SourceRegistry, router: ProviderRouter,
                 max_limit: int = 50):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.state_dir = Path(state_dir).expanduser().resolve()
        self._db = db
        self._manifest = manifest
        self.registry = registry
        self.router = router
        self.max_limit = max_limit

    @classmethod
    def open(cls, data_dir: str | Path, state_dir: str | Path | None = None, *,
             registry: SourceRegistry | None = None,
             pkulaw_call_tool: Callable[..., Any] | None = None,
             pkulaw_provider: PkulawMcpProvider | None = None,
             pkulaw_service_id: str | None = None,
             build_if_missing: bool = False, max_limit: int = 50,
             max_record_bytes: int = 12 * 1024 * 1024,
             max_part_chars: int = 200_000,
             max_asset_bytes: int = 12 * 1024 * 1024,
             max_part_record_bytes: int = 256 * 1024 * 1024) -> "CorpusEnv":
        data_path = Path(data_dir).expanduser().resolve()
        if not data_path.is_dir():
            raise EnvironmentNotReadyError(f"数据目录不存在: {data_path}")
        state_path = Path(state_dir or data_path.parent / ".lexverse").expanduser().resolve()
        try:
            state_path.relative_to(data_path)
        except ValueError:
            pass
        else:
            raise InvalidArgumentError("state_dir 不能位于 data_dir 内部")
        chosen_registry = registry or SourceRegistry.default()
        if build_if_missing and (not (state_path / "catalog.sqlite").is_file() or
                                  not (state_path / "manifest.json").is_file()):
            IndexBuilder(data_path, state_path, registry=chosen_registry).build()
        db = open_readonly_database(state_path)
        try:
            manifest = load_manifest(state_path)
            query = QueryService(
                db,
                allowed_collections=set(chosen_registry.collections()),
                max_limit=max_limit,
            )
            expected_versions = {
                "schema_version": SCHEMA_VERSION,
                "loader_version": LOADER_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
            }
            if any(
                manifest.get(name) != expected or query.meta.get(name) != expected
                for name, expected in expected_versions.items()
            ):
                raise EnvironmentNotReadyError(
                    "索引版本与当前环境不兼容，请重新构建索引"
                )
            if manifest.get("sources") != chosen_registry.describe():
                raise EnvironmentNotReadyError(
                    "索引来源注册表与当前环境不一致，请重新构建索引"
                )
            if manifest.get("index_generation") != query.meta.get("index_generation"):
                raise EnvironmentNotReadyError(
                    "manifest 与索引版本不一致，请重新构建索引"
                )
            if manifest.get("snapshot_id") != compute_snapshot_id(data_path):
                raise EnvironmentNotReadyError("数据目录已发生变化，请重新构建索引")
            store = RecordStore(
                db,
                data_path,
                max_record_bytes=max_record_bytes,
                max_part_chars=max_part_chars,
                max_asset_bytes=max_asset_bytes,
                max_part_record_bytes=max_part_record_bytes,
            )
            local = LocalProvider(query, store)
            external = pkulaw_provider or PkulawMcpProvider(
                pkulaw_call_tool,
                service_id=pkulaw_service_id
                or os.environ.get("LEXVERSE_PKULAW_SERVICE_ID"),
            )
            return cls(
                data_dir=data_path,
                state_dir=state_path,
                db=db,
                manifest=manifest,
                registry=chosen_registry,
                router=ProviderRouter((local, external)),
                max_limit=max_limit,
            )
        except Exception:
            db.close()
            raise

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None  # type: ignore[assignment]

    def __enter__(self) -> "CorpusEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def get_manifest(self) -> dict[str, Any]:
        manifest = deepcopy(self._manifest)
        manifest.pop("data_root_absolute", None)
        manifest["providers_available"] = list(self.router.available_names())
        manifest["providers_registered"] = list(self.router.names())
        return manifest

    def list_collections(self) -> dict[str, Any]:
        collections: list[dict[str, Any]] = []
        manifest_collections = self._manifest.get("collections", {})
        for collection in self.registry.collections():
            details = manifest_collections.get(collection, {})
            providers = list(details.get("providers", ["local"]))
            for provider in self.router.available_names():
                if provider != "local" and provider not in providers:
                    providers.append(provider)
            collections.append({
                "name": collection,
                "label": details.get("label", public_collection_label(collection)),
                "providers": providers,
                "record_count": details.get("record_count"),
                "capabilities": list(details.get("capabilities", ["search", "get"])),
            })
        return {"collections": collections}

    def describe_collection(self, collection: str, provider: str = "local") -> dict[str, Any]:
        if not isinstance(collection, str) or not collection:
            raise InvalidArgumentError("collection 必须是非空字符串")
        if not isinstance(provider, str) or not provider:
            raise InvalidArgumentError("provider 必须是非空字符串")
        if collection not in self.registry.collections():
            raise UnknownCollectionError(f"未知集合: {collection}")
        self.router.get(provider)
        sources = ["pkulaw"] if provider == "pkulaw" else sorted({
            definition.source for definition in self.registry.for_collection(collection)
        })
        details = self._manifest.get("collections", {}).get(collection, {})
        fields = list(FILTER_FIELDS_BY_COLLECTION.get(collection, ("source",)))
        search_fields = {
            "legal_laws": ["law_name", "article", "text"],
            "legal_cases": ["title", "case_number", "case_cause", "court", "facts"],
            "legal_qa": ["question", "topic", "content"],
            "legal_concepts": ["term", "description", "facts"],
            "legal_templates": ["title", "template_type", "case_type", "content"],
        }.get(collection, [])
        capabilities = list(details.get("capabilities", ["search", "get", "read_part"]))
        if provider != "local":
            capabilities = ["search", "get"]
            fields = []
            search_fields = []
        return {
            "collection": collection,
            "provider": provider,
            "sources": sources,
            "filter_fields": fields,
            "search_fields": search_fields,
            "capabilities": capabilities,
            "raw_schema": "source-specific" if provider == "local" else "provider-specific",
        }

    def search_records(self, collection: str, query: str | None = None,
                       filters: dict[str, Any] | None = None,
                       sort: str = "relevance", limit: int = 10,
                       cursor: str | None = None, provider: str = "local") -> dict[str, Any]:
        if not isinstance(collection, str) or not collection:
            raise InvalidArgumentError("collection 必须是非空字符串")
        if not isinstance(provider, str) or not provider:
            raise InvalidArgumentError("provider 必须是非空字符串")
        if collection not in self.registry.collections():
            raise UnknownCollectionError(f"未知集合: {collection}")
        self.router.get(provider)
        if query is not None and not isinstance(query, str):
            raise InvalidArgumentError("query 必须是字符串或 null")
        if filters is not None and not isinstance(filters, dict):
            raise InvalidArgumentError("filters 必须是对象")
        if not isinstance(sort, str) or sort not in {"relevance", "stable"}:
            raise InvalidArgumentError("sort 只能是 relevance 或 stable")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise InvalidArgumentError("cursor 必须是非空字符串或 null")
        limit = validate_limit(limit, maximum=self.max_limit)
        return self.router.search(SearchRequest(
            collection=collection, provider=provider, query=query,
            filters=filters, sort=sort, limit=limit, cursor=cursor,
        ))

    def get_record(self, record_id: str) -> dict[str, Any]:
        if not isinstance(record_id, str) or not record_id:
            raise InvalidArgumentError("record_id 必须是非空字符串")
        return self.router.get_record(record_id)

    def read_record_part(self, record_id: str, path: str,
                         offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        if not isinstance(record_id, str) or not record_id:
            raise InvalidArgumentError("record_id 必须是非空字符串")
        return self.router.read_record_part(record_id, path, offset=offset, limit=limit)

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        return self.router.get_asset(asset_id, mode=mode)
