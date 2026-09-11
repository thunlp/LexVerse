"""SQLite catalog and FTS index builder for the local LexVerse corpus."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .errors import EnvironmentNotReadyError, InvalidArgumentError, error_response
from .registry import SourceRegistry, collection_short_name, public_collection_label
from .utils import safe_path, stable_hash, tokenize_for_index

SCHEMA_VERSION = "0.1"
LOADER_VERSION = "0.3"
TOKENIZER_VERSION = "nfkc-cjk-unigram-bigram-v2"
SNAPSHOT_METHOD = "content-manifest-sha256-v1"


def _nonempty(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidArgumentError(f"{name} 必须是非空字符串")
    return value.strip()


def make_data_source(
    *,
    source_type: str,
    file_manifest_digest: str,
    hf_repo_id: str | None = None,
    huggingface_commit_sha: str | None = None,
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
) -> dict[str, str]:
    if source_type == "huggingface":
        if dataset_id is not None:
            raise InvalidArgumentError("Hugging Face 数据源不能设置 dataset_id")
        repo_id = _nonempty(hf_repo_id, "hf_repo_id")
        commit_sha = _nonempty(huggingface_commit_sha, "huggingface_commit_sha")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit_sha):
            raise InvalidArgumentError(
                "huggingface_commit_sha 必须是 40 位十六进制 commit SHA"
            )
        return {
            "type": "huggingface",
            "repo_id": repo_id,
            "commit_sha": commit_sha.lower(),
            "revision": _nonempty(dataset_revision, "dataset_revision"),
        }
    if source_type == "local":
        if hf_repo_id is not None or huggingface_commit_sha is not None:
            raise InvalidArgumentError("本地数据源不能设置 Hugging Face 参数")
        local_dataset_id = "local" if dataset_id is None else _nonempty(
            dataset_id, "dataset_id"
        )
        revision = (
            f"content-{file_manifest_digest[:16]}"
            if dataset_revision is None
            else _nonempty(dataset_revision, "dataset_revision")
        )
        return {
            "type": "local",
            "dataset_id": local_dataset_id,
            "revision": revision,
        }
    raise InvalidArgumentError("source_type 只能是 huggingface 或 local")


def make_public_id(collection: str, source: str, source_key: str) -> str:
    key = str(source_key).replace("\r", " ").replace("\n", " ").strip() or "anonymous"
    return f"lv:{collection_short_name(collection)}:{source}:{key}"


def _snapshot_files(data_root: str | Path) -> list[tuple[str, Path]]:
    data_root = Path(data_root).expanduser().resolve()
    files: list[tuple[str, Path]] = []
    for candidate in data_root.rglob("*"):
        relative_file = candidate.relative_to(data_root).as_posix()
        # huggingface-cli may create mutable download bookkeeping under the
        # destination. It is not part of the dataset snapshot.
        if (
            relative_file in {".gitkeep", ".DS_Store", ".cache"}
            or relative_file.startswith(".cache/huggingface/")
        ):
            continue
        path = safe_path(data_root, relative_file)
        if path.is_file():
            files.append((relative_file, path))
    return sorted(files)


def build_file_manifest(data_root: str | Path, *, hash_content: bool = True) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for relative_file, path in _snapshot_files(data_root):
        size = path.stat().st_size
        entry: dict[str, Any] = {"relative_path": relative_file, "size": size}
        if hash_content:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        entries.append(entry)
        total_bytes += size
    manifest: dict[str, Any] = {
        "algorithm": "sha256",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    if hash_content:
        manifest["digest"] = stable_hash(entries, length=64)
    return manifest


def file_stat_signature(data_root: str | Path) -> str:
    """Cheap mutable-data signal; timestamps are not part of snapshot identity."""

    entries: list[dict[str, Any]] = []
    try:
        for relative_file, path in _snapshot_files(data_root):
            stat = path.stat()
            entries.append({
                "relative_path": relative_file,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            })
    except OSError as exc:
        raise EnvironmentNotReadyError("用户数据文件无法读取，请检查后重建用户索引") from exc
    return stable_hash(entries, length=64)


def summarize_file_manifest(file_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        name: file_manifest.get(name)
        for name in ("algorithm", "digest", "file_count", "total_bytes")
    }


def validate_file_manifest(
    data_root: str | Path,
    expected: dict[str, Any],
    *,
    verify_content: bool = False,
) -> None:
    expected_files = expected.get("files") if isinstance(expected, dict) else None
    if not isinstance(expected_files, list) or expected.get("algorithm") != "sha256":
        raise EnvironmentNotReadyError("索引中的数据文件清单无效，请重新构建索引")
    valid_entries = all(
        isinstance(entry, dict)
        and isinstance(entry.get("relative_path"), str)
        and bool(entry["relative_path"])
        and isinstance(entry.get("size"), int)
        and not isinstance(entry["size"], bool)
        and entry["size"] >= 0
        and isinstance(entry.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None
        for entry in expected_files
    )
    if (
        not valid_entries
        or expected.get("file_count") != len(expected_files)
        or expected.get("total_bytes") != sum(entry["size"] for entry in expected_files)
        or expected.get("digest") != stable_hash(expected_files, length=64)
    ):
        raise EnvironmentNotReadyError("索引中的数据文件清单校验失败，请重新获取索引")
    actual = build_file_manifest(data_root, hash_content=verify_content)
    fields = ("relative_path", "size", "sha256") if verify_content else (
        "relative_path",
        "size",
    )
    expected_projection = [
        {name: entry.get(name) for name in fields}
        for entry in expected_files
        if isinstance(entry, dict)
    ]
    actual_projection = [
        {name: entry.get(name) for name in fields} for entry in actual["files"]
    ]
    if len(expected_projection) != len(expected_files) or actual_projection != expected_projection:
        detail = "内容" if verify_content else "路径或大小"
        raise EnvironmentNotReadyError(f"数据文件{detail}与索引不一致，请重新构建索引")


def compute_snapshot_id(*, data_source: dict[str, str], file_manifest: dict[str, Any]) -> str:
    """Fingerprint source provenance and a complete SHA-256 file manifest."""

    return "snapshot-" + stable_hash(
        {
            "method": SNAPSHOT_METHOD,
            "data_source": data_source,
            "file_manifest_digest": file_manifest.get("digest"),
        },
        length=24,
    )


def _create_schema(db: sqlite3.Connection) -> bool:
    db.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE records(
            id TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            provider TEXT NOT NULL,
            source TEXT NOT NULL,
            source_key TEXT NOT NULL,
            key_fields_json TEXT NOT NULL,
            filter_fields_json TEXT NOT NULL,
            locator_json TEXT NOT NULL,
            search_text TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX records_collection_provider_id
            ON records(collection, provider, id);
        CREATE INDEX records_source ON records(collection, source);
        CREATE TABLE assets(
            asset_id TEXT PRIMARY KEY,
            record_id TEXT NOT NULL,
            relative_file TEXT NOT NULL,
            mime_type TEXT,
            byte_size INTEGER NOT NULL,
            FOREIGN KEY(record_id) REFERENCES records(id)
        );
        CREATE INDEX assets_record_id ON assets(record_id);
        """
    )
    try:
        db.execute(
            "CREATE VIRTUAL TABLE records_fts USING fts5("
            "id UNINDEXED, search_text, tokenize='unicode61')"
        )
        return True
    except sqlite3.OperationalError:
        return False


def _write_meta(db: sqlite3.Connection, values: dict[str, Any]) -> None:
    db.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in values.items()],
    )


class IndexBuilder:
    def __init__(
        self,
        data_root: str | Path,
        state_dir: str | Path,
        *,
        registry: SourceRegistry | None = None,
        provider: str = "local",
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.registry = registry or SourceRegistry.default()
        self.provider = provider
        if not self.data_root.is_dir():
            raise EnvironmentNotReadyError(f"数据目录不存在: {self.data_root}")
        try:
            self.state_dir.relative_to(self.data_root)
        except ValueError:
            pass
        else:
            raise InvalidArgumentError("state_dir 不能位于 data_dir 内部")

    @staticmethod
    def _asset_id(source: str, relative_file: str) -> str:
        display = relative_file.removeprefix("legal_templates/")
        return f"lv:asset:legal_templates:{source}:{display}"

    def build(
        self,
        *,
        source_type: str = "local",
        hf_repo_id: str | None = None,
        huggingface_commit_sha: str | None = None,
        dataset_id: str | None = None,
        dataset_revision: str | None = None,
        max_records: int | None = None,
        sources: set[str] | None = None,
        progress: bool = False,
    ) -> dict[str, Any]:
        if max_records is not None and (
            isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1
        ):
            raise InvalidArgumentError("max_records 必须是正整数或 null")
        # Hash once while building. Normal opens use the stored path/size
        # projection; doctor --verify-data checks these content hashes again.
        file_manifest = build_file_manifest(self.data_root)
        data_source = make_data_source(
            source_type=source_type,
            file_manifest_digest=file_manifest["digest"],
            hf_repo_id=hf_repo_id,
            huggingface_commit_sha=huggingface_commit_sha,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
        )
        snapshot_id = compute_snapshot_id(
            data_source=data_source,
            file_manifest=file_manifest,
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        final_db = self.state_dir / "catalog.sqlite"
        temp_db = self.state_dir / f"catalog.sqlite.tmp-{os.getpid()}"
        if temp_db.exists():
            temp_db.unlink()
        db = sqlite3.connect(temp_db)
        try:
            db.execute("PRAGMA synchronous=NORMAL")
            has_fts = _create_schema(db)
        except Exception:
            db.close()
            if temp_db.exists():
                temp_db.unlink()
            raise
        seen_ids: set[str] = set()
        collection_counts: Counter[str] = Counter()
        source_counts: Counter[tuple[str, str]] = Counter()
        source_names: defaultdict[str, set[str]] = defaultdict(set)
        errors: list[dict[str, Any]] = []
        record_batch: list[tuple[Any, ...]] = []
        fts_batch: list[tuple[str, str]] = []
        asset_batch: list[tuple[Any, ...]] = []
        total = 0

        def flush() -> None:
            if record_batch:
                db.executemany(
                    """
                    INSERT INTO records(
                        id, collection, provider, source, source_key,
                        key_fields_json, filter_fields_json, locator_json, search_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    record_batch,
                )
                record_batch.clear()
            if has_fts and fts_batch:
                db.executemany(
                    "INSERT INTO records_fts(id, search_text) VALUES (?, ?)",
                    fts_batch,
                )
                fts_batch.clear()
            if asset_batch:
                db.executemany(
                    """
                    INSERT OR REPLACE INTO assets(
                        asset_id, record_id, relative_file, mime_type, byte_size
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    asset_batch,
                )
                asset_batch.clear()
            db.commit()

        try:
            for definition in self.registry.definitions():
                if sources and definition.source not in sources:
                    continue
                loader = definition.build_loader()
                for loaded in loader.iter_records(self.data_root):
                    if max_records is not None and total >= max_records:
                        break
                    base_id = make_public_id(
                        loaded.collection, loaded.source, loaded.source_key
                    )
                    record_id = base_id
                    suffix = 2
                    while record_id in seen_ids:
                        record_id = f"{base_id}~{suffix}"
                        suffix += 1
                    seen_ids.add(record_id)
                    record_batch.append(
                        (
                            record_id,
                            loaded.collection,
                            self.provider,
                            loaded.source,
                            loaded.source_key,
                            json.dumps(
                                loaded.key_fields,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            json.dumps(
                                loaded.filter_fields,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            json.dumps(
                                loaded.locator,
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            ),
                            "" if has_fts else loaded.search_text[:120000],
                        )
                    )
                    if has_fts:
                        fts_batch.append(
                            (record_id, tokenize_for_index(loaded.search_text))
                        )
                    collection_counts[loaded.collection] += 1
                    source_counts[(loaded.collection, loaded.source)] += 1
                    source_names[loaded.collection].add(loaded.source)
                    for relative_file in loaded.asset_paths:
                        try:
                            path = safe_path(
                                self.data_root,
                                relative_file,
                                allow_suffixes={".docx", ".doc", ".pdf", ".txt"},
                            )
                            asset_batch.append(
                                (
                                    self._asset_id(loaded.source, relative_file),
                                    record_id,
                                    relative_file,
                                    mimetypes.guess_type(path.name)[0],
                                    path.stat().st_size,
                                )
                            )
                        except (OSError, ValueError) as exc:
                            errors.append(
                                {
                                    "source": loaded.source,
                                    "file": relative_file,
                                    **error_response(exc),
                                }
                            )
                    total += 1
                    if progress and total % 1000 == 0:
                        print(f"indexed {total} records", file=sys.stderr)
                    if len(record_batch) >= 500:
                        flush()
                if max_records is not None and total >= max_records:
                    break
            flush()
            index_generation = stable_hash(
                {
                    "schema": SCHEMA_VERSION,
                    "loader": LOADER_VERSION,
                    "tokenizer": TOKENIZER_VERSION,
                    "registry": self.registry.describe(),
                    "snapshot": snapshot_id,
                    "provider": self.provider,
                    "counts": {
                        f"{collection}:{source}": count
                        for (collection, source), count in sorted(source_counts.items())
                    },
                    "fts": has_fts,
                },
                length=32,
            )
            _write_meta(
                db,
                {
                    "schema_version": SCHEMA_VERSION,
                    "loader_version": LOADER_VERSION,
                    "tokenizer_version": TOKENIZER_VERSION,
                    "snapshot_id": snapshot_id,
                    "snapshot_method": SNAPSHOT_METHOD,
                    "data_source": data_source,
                    "file_manifest_digest": file_manifest["digest"],
                    "index_generation": index_generation,
                    "provider": self.provider,
                    "has_fts": has_fts,
                    "partial": max_records is not None,
                    "total_records": total,
                },
            )
            db.commit()
            db.close()
            os.replace(temp_db, final_db)
            return self._write_manifest(
                snapshot_id=snapshot_id,
                data_source=data_source,
                file_manifest=file_manifest,
                index_generation=index_generation,
                collection_counts=collection_counts,
                source_counts=source_counts,
                source_names=source_names,
                errors=errors,
                partial=max_records is not None,
            )
        except Exception:
            db.close()
            if temp_db.exists():
                temp_db.unlink()
            raise

    def _write_manifest(
        self,
        *,
        snapshot_id: str,
        data_source: dict[str, str],
        file_manifest: dict[str, Any],
        index_generation: str,
        collection_counts: Counter[str],
        source_counts: Counter[tuple[str, str]],
        source_names: defaultdict[str, set[str]],
        errors: list[dict[str, Any]],
        partial: bool,
    ) -> dict[str, Any]:
        collections: dict[str, Any] = {}
        for collection in self.registry.collections():
            if collection not in source_names:
                continue
            collections[collection] = {
                "label": public_collection_label(collection),
                "record_count": collection_counts[collection],
                "sources": sorted(source_names[collection]),
                "source_record_counts": {
                    source: source_counts[(collection, source)]
                    for source in sorted(source_names[collection])
                },
                "providers": [self.provider],
                "capabilities": ["search", "get", "read_part"],
            }
            if collection == "legal_templates":
                collections[collection]["capabilities"].append("asset")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "loader_version": LOADER_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "snapshot_method": SNAPSHOT_METHOD,
            "snapshot_id": snapshot_id,
            "data_source": data_source,
            "file_manifest": file_manifest,
            "index_generation": index_generation,
            "provider": self.provider,
            "data_root": "data",
            "partial": partial,
            "collections": collections,
            "sources": self.registry.describe(),
            "errors": errors,
        }
        temp_manifest = self.state_dir / f"manifest.json.tmp-{os.getpid()}"
        with temp_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_manifest, self.state_dir / "manifest.json")
        return manifest


def read_meta(db: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in db.execute("SELECT key, value FROM meta"):
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def open_readonly_database(state_dir: str | Path) -> sqlite3.Connection:
    path = Path(state_dir).expanduser().resolve() / "catalog.sqlite"
    if not path.is_file():
        raise EnvironmentNotReadyError(
            f"索引不存在: {path}；请先运行 python -m lexverse_env.cli build-index"
        )
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def load_manifest(state_dir: str | Path) -> dict[str, Any]:
    path = Path(state_dir).expanduser().resolve() / "manifest.json"
    if not path.is_file():
        raise EnvironmentNotReadyError(f"manifest 不存在: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentNotReadyError("manifest 无法读取或解析，请重新构建索引") from exc
    if not isinstance(manifest, dict):
        raise EnvironmentNotReadyError("manifest 结构无效，请重新构建索引")
    return manifest
