"""Command-line utilities for building and inspecting LexVerse."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .env import CorpusEnv
from .errors import EnvironmentNotReadyError, LexVerseError, error_response
from .index import IndexBuilder, summarize_file_manifest
from .registry import SourceRegistry


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)


def _user_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user-data-dir", type=Path, default=None)
    parser.add_argument("--user-state-dir", type=Path, default=None)


def _top_level_type(path: Path) -> str:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    try:
        with path.open("rb") as handle:
            prefix = handle.read(4096).lstrip()
    except OSError:
        return "unreadable"
    if not prefix:
        return "empty"
    if prefix.startswith(b"["):
        return "array"
    if prefix.startswith(b"{"):
        return "object"
    return "other"


def command_inventory(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise EnvironmentNotReadyError(f"数据目录不存在: {data_dir}")
    registry = SourceRegistry.default()
    file_counts: Counter[tuple[str, str]] = Counter()
    byte_counts: Counter[tuple[str, str]] = Counter()
    top_levels: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    record_counts: Counter[tuple[str, str]] = Counter()
    errors: list[dict[str, Any]] = []
    definitions_seen: set[tuple[str, str, str, str]] = set()
    for definition in registry.definitions():
        marker = (definition.collection, definition.source, definition.loader_kind, definition.pattern)
        if marker in definitions_seen:
            continue
        definitions_seen.add(marker)
        key = (definition.collection, definition.source)
        loader = definition.build_loader()
        paths = sorted(
            (path for path in data_dir.glob(definition.pattern) if path.is_file()),
            key=lambda path: path.as_posix(),
        )
        for path in paths:
            file_counts[key] += 1
            try:
                byte_counts[key] += path.stat().st_size
            except OSError:
                pass
            top_levels[key][_top_level_type(path)] += 1
        if args.count_records:
            try:
                for number, _record in enumerate(loader.iter_records(data_dir), start=1):
                    record_counts[key] += 1
                    if args.max_records and number >= args.max_records:
                        break
            except Exception as exc:
                errors.append({
                    "collection": definition.collection,
                    "source": definition.source,
                    **error_response(exc),
                })
    sources: list[dict[str, Any]] = []
    for key in sorted(file_counts):
        collection, source = key
        item: dict[str, Any] = {
            "collection": collection,
            "source": source,
            "file_count": file_counts[key],
            "byte_size": byte_counts[key],
            "top_level_types": dict(sorted(top_levels[key].items())),
        }
        if args.count_records:
            item["record_count"] = record_counts[key]
        sources.append(item)
    result: dict[str, Any] = {"data_dir": "data", "sources": sources, "errors": errors}
    if args.count_records and args.max_records:
        result["record_count_limit"] = args.max_records
    _json_dump(result)
    return 0 if not errors else 2


def command_build_index(args: argparse.Namespace) -> int:
    result = IndexBuilder(args.data_dir, args.state_dir, registry=SourceRegistry.default()).build(
        source_type=args.source_type,
        hf_repo_id=args.hf_repo_id,
        huggingface_commit_sha=args.huggingface_commit_sha,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        max_records=args.max_records,
        progress=args.progress,
    )
    file_manifest = result.get("file_manifest")
    if isinstance(file_manifest, dict):
        result["file_manifest"] = summarize_file_manifest(file_manifest)
    _json_dump(result)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    env = CorpusEnv.open(
        args.data_dir,
        args.state_dir,
        user_data_dir=args.user_data_dir,
        user_state_dir=args.user_state_dir,
        max_limit=args.max_limit,
        verify_data=args.verify_data,
    )
    try:
        manifest = env.get_manifest()
        collections = env.list_collections()
        checks: dict[str, Any] = {
            "data_dir": "data",
            "state_dir": ".lexverse",
            "data_verified": args.verify_data,
            "collection_count": len(collections.get("collections", [])),
            "providers": list(env.router.available_names()),
            "ok": True,
        }
        if manifest.get("layout") == "layered":
            checks["layers"] = {
                name: {
                    "snapshot_id": layer.get("snapshot_id"),
                    "data_source": layer.get("data_source"),
                    "index_generation": layer.get("index_generation"),
                }
                for name, layer in manifest.get("layers", {}).items()
            }
        else:
            checks.update({
                "snapshot_id": manifest.get("snapshot_id"),
                "data_source": manifest.get("data_source"),
                "index_generation": manifest.get("index_generation"),
            })
        for item in collections.get("collections", []):
            if item.get("record_count", 0):
                page = env.search_records(item["name"], sort="stable", limit=1)
                if page.get("items"):
                    sample_id = page["items"][0]["id"]
                    sample = env.get_record(sample_id)
                    if "record" not in sample or "provenance" not in sample:
                        raise LexVerseError("记录抽样读取响应不完整")
                    checks.setdefault("sample_ids", {})[item["name"]] = sample_id
        _json_dump(checks)
        return 0
    finally:
        env.close()


def command_search(args: argparse.Namespace) -> int:
    env = CorpusEnv.open(
        args.data_dir,
        args.state_dir,
        user_data_dir=args.user_data_dir,
        user_state_dir=args.user_state_dir,
        max_limit=args.max_limit,
    )
    try:
        filters = json.loads(args.filters) if args.filters else None
        _json_dump(env.search_records(collection=args.collection, provider=args.provider,
                                      query=args.query, filters=filters, sort=args.sort,
                                      limit=args.limit, cursor=args.cursor))
        return 0
    finally:
        env.close()


def command_get(args: argparse.Namespace) -> int:
    env = CorpusEnv.open(
        args.data_dir,
        args.state_dir,
        user_data_dir=args.user_data_dir,
        user_state_dir=args.user_state_dir,
        max_limit=args.max_limit,
    )
    try:
        if args.asset:
            result = env.get_asset(args.identifier, mode=args.mode)
        elif args.part:
            result = env.read_record_part(args.identifier, args.part, offset=args.offset, limit=args.limit)
        else:
            result = env.get_record(args.identifier)
        _json_dump(result)
        return 0
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexverse-env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="扫描已注册来源")
    _path_args(inventory)
    inventory.add_argument("--count-records", action="store_true")
    inventory.add_argument("--max-records", type=int, default=None)
    inventory.set_defaults(handler=command_inventory)

    build = subparsers.add_parser("build-index", help="构建只读 SQLite/FTS 索引")
    _path_args(build)
    build.add_argument(
        "--source-type",
        choices=("local", "huggingface"),
        default="local",
        help="数据来源类型（默认: local）",
    )
    build.add_argument("--hf-repo-id", default=None, help="Hugging Face 数据集仓库 ID")
    build.add_argument(
        "--huggingface-commit-sha",
        default=None,
        help="数据下载所对应的 40 位 Hugging Face commit SHA",
    )
    build.add_argument("--dataset-id", default=None, help="本地数据集标识")
    build.add_argument(
        "--dataset-revision",
        default=None,
        help="数据集 revision；本地数据省略时根据内容生成",
    )
    build.add_argument("--max-records", type=int, default=None)
    build.add_argument("--progress", action="store_true")
    build.set_defaults(handler=command_build_index)

    doctor = subparsers.add_parser("doctor", help="检查索引和环境")
    _path_args(doctor)
    _user_path_args(doctor)
    doctor.add_argument("--max-limit", type=int, default=50)
    doctor.add_argument(
        "--verify-data",
        action="store_true",
        help="重新计算所有数据文件的 SHA-256",
    )
    doctor.set_defaults(handler=command_doctor)

    search = subparsers.add_parser("search", help="搜索记录引用")
    _path_args(search)
    _user_path_args(search)
    search.add_argument("--collection", required=True)
    search.add_argument("--provider", default="local")
    search.add_argument("--query", default=None)
    search.add_argument("--filters", default=None, help="JSON object")
    search.add_argument("--sort", choices=("relevance", "stable"), default="relevance")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--cursor", default=None)
    search.add_argument("--max-limit", type=int, default=50)
    search.set_defaults(handler=command_search)

    get = subparsers.add_parser("get", help="读取记录、片段或资产")
    _path_args(get)
    _user_path_args(get)
    get.add_argument("identifier")
    get.add_argument("--part", default=None, help="JSON Pointer")
    get.add_argument("--offset", type=int, default=0)
    get.add_argument("--limit", type=int, default=12_000)
    get.add_argument("--asset", action="store_true")
    get.add_argument("--mode", choices=("metadata", "text", "binary"), default="metadata")
    get.add_argument("--max-limit", type=int, default=50)
    get.set_defaults(handler=command_get)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:
        _json_dump(error_response(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
