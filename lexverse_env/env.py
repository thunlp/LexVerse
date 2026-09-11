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
    SNAPSHOT_METHOD,
    TOKENIZER_VERSION,
    compute_snapshot_id,
    file_stat_signature,
    load_manifest,
    open_readonly_database,
    summarize_file_manifest,
    validate_file_manifest,
)
from .providers import (
    LayeredLocalProvider,
    LocalProvider,
    PkulawMcpProvider,
    ProviderRouter,
)
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
        self._extra_envs: list[CorpusEnv] = []
        self._layer_manifests: dict[str, dict[str, Any]] = {"official": manifest}
        self._mutable_layers: list[dict[str, Any]] = []

    @classmethod
    def open(cls, data_dir: str | Path, state_dir: str | Path | None = None, *,
             registry: SourceRegistry | None = None,
             pkulaw_call_tool: Callable[..., Any] | None = None,
             pkulaw_provider: PkulawMcpProvider | None = None,
             pkulaw_service_id: str | None = None,
             pkulaw_token: str | None = None,
             pkulaw_endpoint: str | None = None,
             pkulaw_endpoints: dict[str, str] | None = None,
             pkulaw_timeout_seconds: float = 30.0,
             build_if_missing: bool = False,
             source_type: str = "local",
             hf_repo_id: str | None = None,
             huggingface_commit_sha: str | None = None,
             dataset_id: str | None = None,
             dataset_revision: str | None = None,
             verify_data: bool = False,
             user_data_dir: str | Path | None = None,
             user_state_dir: str | Path | None = None,
             _load_user_layer: bool = True,
             max_limit: int = 50,
             max_record_bytes: int = 12 * 1024 * 1024,
             max_part_chars: int = 200_000,
             max_asset_bytes: int = 12 * 1024 * 1024,
             max_part_record_bytes: int = 256 * 1024 * 1024) -> "CorpusEnv":
        data_path = Path(data_dir).expanduser().resolve()
        if not data_path.is_dir():
            raise EnvironmentNotReadyError(f"数据目录不存在: {data_path}")
        state_root = Path(state_dir or data_path.parent / ".lexverse").expanduser().resolve()
        state_path = (
            state_root / "official"
            if (state_root / "official" / "catalog.sqlite").is_file()
            else state_root
        )
        state_container = (
            state_root.parent
            if state_path == state_root and state_root.name == "official"
            else state_root
        )
        try:
            state_path.relative_to(data_path)
        except ValueError:
            pass
        else:
            raise InvalidArgumentError("state_dir 不能位于 data_dir 内部")
        chosen_registry = registry or SourceRegistry.default()
        if build_if_missing and (not (state_path / "catalog.sqlite").is_file() or
                                  not (state_path / "manifest.json").is_file()):
            IndexBuilder(data_path, state_path, registry=chosen_registry).build(
                source_type=source_type,
                hf_repo_id=hf_repo_id,
                huggingface_commit_sha=huggingface_commit_sha,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
            )
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
            indexed_sources = manifest.get("sources")
            registered_sources = chosen_registry.describe()
            if (
                not isinstance(indexed_sources, list)
                or any(source not in registered_sources for source in indexed_sources)
            ):
                raise EnvironmentNotReadyError(
                    "索引来源注册表与当前环境不一致，请重新构建索引"
                )
            if manifest.get("index_generation") != query.meta.get("index_generation"):
                raise EnvironmentNotReadyError(
                    "manifest 与索引版本不一致，请重新构建索引"
                )
            identity_fields = (
                "snapshot_id",
                "snapshot_method",
                "data_source",
            )
            if manifest.get("snapshot_method") != SNAPSHOT_METHOD or any(
                manifest.get(name) != query.meta.get(name) for name in identity_fields
            ):
                raise EnvironmentNotReadyError(
                    "索引的数据集身份信息缺失或不一致，请重新构建索引"
                )
            file_manifest = manifest.get("file_manifest")
            if (
                not isinstance(file_manifest, dict)
                or file_manifest.get("digest") != query.meta.get("file_manifest_digest")
                or compute_snapshot_id(
                    data_source=manifest["data_source"],
                    file_manifest=file_manifest,
                ) != manifest["snapshot_id"]
            ):
                raise EnvironmentNotReadyError(
                    "manifest 与索引的数据文件清单不一致，请重新构建索引"
                )
            validate_file_manifest(
                data_path,
                file_manifest,
                verify_content=verify_data,
            )
            store = RecordStore(
                db,
                data_path,
                max_record_bytes=max_record_bytes,
                max_part_chars=max_part_chars,
                max_asset_bytes=max_asset_bytes,
                max_part_record_bytes=max_part_record_bytes,
            )
            local = LocalProvider(query, store)
            service_id = (
                pkulaw_service_id
                or os.environ.get("LEXVERSE_PKULAW_SERVICE_ID")
            )
            if pkulaw_provider is not None:
                external = pkulaw_provider
            elif pkulaw_call_tool is not None:
                external = PkulawMcpProvider.standard(
                    call_tool=pkulaw_call_tool,
                    service_id=service_id,
                    endpoint=pkulaw_endpoint,
                    endpoints=pkulaw_endpoints,
                    timeout_seconds=pkulaw_timeout_seconds,
                )
            else:
                external = PkulawMcpProvider.from_env(
                    token=pkulaw_token,
                    endpoint=pkulaw_endpoint,
                    endpoints=pkulaw_endpoints,
                    service_id=service_id,
                    timeout_seconds=pkulaw_timeout_seconds,
                )
            result = cls(
                data_dir=data_path,
                state_dir=state_path,
                db=db,
                manifest=manifest,
                registry=chosen_registry,
                router=ProviderRouter((local, external)),
                max_limit=max_limit,
            )
            if _load_user_layer:
                configured_user_data = (
                    Path(user_data_dir).expanduser().resolve()
                    if user_data_dir is not None
                    else data_path.parent / "user_data"
                )
                configured_user_state = (
                    Path(user_state_dir).expanduser().resolve()
                    if user_state_dir is not None
                    else state_container / "user"
                )
                user_index_exists = (
                    (configured_user_state / "catalog.sqlite").is_file()
                    and (configured_user_state / "manifest.json").is_file()
                )
                user_data_exists = configured_user_data.is_dir()
                if user_data_exists and not user_index_exists:
                    raise EnvironmentNotReadyError(
                        "检测到用户数据但用户索引不存在，请仅为 user_data 构建用户索引"
                    )
                if user_index_exists and not user_data_exists:
                    raise EnvironmentNotReadyError(
                        "检测到用户索引但用户数据目录不存在"
                    )
                if user_data_dir is not None and not user_data_exists:
                    raise EnvironmentNotReadyError(
                        f"用户数据目录不存在: {configured_user_data}"
                    )
                if user_state_dir is not None and not user_index_exists:
                    raise EnvironmentNotReadyError(
                        f"用户索引不存在或不完整: {configured_user_state}"
                    )
                if user_data_exists and user_index_exists:
                    user_env = cls.open(
                        configured_user_data,
                        configured_user_state,
                        registry=chosen_registry,
                        # User data is mutable and normally much smaller than
                        # the official corpus, so never trust path/size alone.
                        verify_data=True,
                        max_limit=max_limit,
                        max_record_bytes=max_record_bytes,
                        max_part_chars=max_part_chars,
                        max_asset_bytes=max_asset_bytes,
                        max_part_record_bytes=max_part_record_bytes,
                        _load_user_layer=False,
                    )
                    user_local = user_env.router.get("local")
                    layered_local = LayeredLocalProvider(
                        (("user", user_local), ("official", local)),
                        max_limit=max_limit,
                    )
                    result.router = ProviderRouter((layered_local, external))
                    result._extra_envs.append(user_env)
                    result._layer_manifests["user"] = user_env._manifest
                    result._mutable_layers.append({
                        "data_dir": configured_user_data,
                        "file_manifest": user_env._manifest["file_manifest"],
                        "stat_signature": file_stat_signature(configured_user_data),
                    })
            return result
        except Exception:
            db.close()
            raise

    def close(self) -> None:
        for environment in self._extra_envs:
            environment.close()
        self._extra_envs.clear()
        if self._db is not None:
            self._db.close()
            self._db = None  # type: ignore[assignment]

    def __enter__(self) -> "CorpusEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _validate_mutable_layers(self) -> None:
        for layer in self._mutable_layers:
            current_signature = file_stat_signature(layer["data_dir"])
            if current_signature == layer["stat_signature"]:
                continue
            validate_file_manifest(
                layer["data_dir"],
                layer["file_manifest"],
                verify_content=True,
            )
            layer["stat_signature"] = file_stat_signature(layer["data_dir"])

    def get_manifest(self) -> dict[str, Any]:
        def public_manifest(value: dict[str, Any]) -> dict[str, Any]:
            manifest = deepcopy(value)
            manifest.pop("data_root_absolute", None)
            file_manifest = manifest.get("file_manifest")
            if isinstance(file_manifest, dict):
                manifest["file_manifest"] = summarize_file_manifest(file_manifest)
            return manifest

        if len(self._layer_manifests) == 1:
            manifest = public_manifest(self._manifest)
        else:
            manifest = {
                "layout": "layered",
                "layers": {
                    name: public_manifest(value)
                    for name, value in self._layer_manifests.items()
                },
            }
        manifest["providers_available"] = list(self.router.available_names())
        manifest["providers_registered"] = list(self.router.names())
        return manifest

    def list_collections(self) -> dict[str, Any]:
        collections: list[dict[str, Any]] = []
        for collection in self.registry.collections():
            layer_details = [
                manifest.get("collections", {}).get(collection, {})
                for manifest in self._layer_manifests.values()
            ]
            details = next((item for item in layer_details if item), {})
            providers = list(details.get("providers", ["local"]))
            for provider in self.router.available_names():
                adapter = self.router.get(provider)
                supports = getattr(adapter, "supports_collection", None)
                if (
                    provider != "local"
                    and provider not in providers
                    and (supports is None or supports(collection))
                ):
                    providers.append(provider)
            collections.append({
                "name": collection,
                "label": details.get("label", public_collection_label(collection)),
                "providers": providers,
                "record_count": sum(
                    int(item.get("record_count", 0)) for item in layer_details
                ),
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
            adapter = self.router.get(provider)
            describe = getattr(adapter, "describe_collection", None)
            remote = describe(collection) if describe else {}
            capabilities = list(remote.get("capabilities", ["search", "get"]))
            fields = list(remote.get("filter_fields", []))
            search_fields = list(remote.get("search_fields", []))
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
        if provider == "local":
            self._validate_mutable_layers()
        limit = validate_limit(limit, maximum=self.max_limit)
        return self.router.search(SearchRequest(
            collection=collection, provider=provider, query=query,
            filters=filters, sort=sort, limit=limit, cursor=cursor,
        ))

    def get_record(self, record_id: str) -> dict[str, Any]:
        if not isinstance(record_id, str) or not record_id:
            raise InvalidArgumentError("record_id 必须是非空字符串")
        if not record_id.startswith("lv:external:"):
            self._validate_mutable_layers()
        return self.router.get_record(record_id)

    def read_record_part(self, record_id: str, path: str,
                         offset: int = 0, limit: int = 12_000) -> dict[str, Any]:
        if not isinstance(record_id, str) or not record_id:
            raise InvalidArgumentError("record_id 必须是非空字符串")
        if not record_id.startswith("lv:external:"):
            self._validate_mutable_layers()
        return self.router.read_record_part(record_id, path, offset=offset, limit=limit)

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        self._validate_mutable_layers()
        return self.router.get_asset(asset_id, mode=mode)
