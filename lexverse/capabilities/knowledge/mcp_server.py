from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
from typing import Annotated

from pydantic import Field

from lexverse.capabilities.knowledge.index import KnowledgeIndex
from lexverse.capabilities.knowledge.retrieval import TaskRetriever
from lexverse.runtime.paths import CAPABILITIES_ROOT
from lexverse.tasks.user import KnowledgeFilters, RetrievalConfig


def _load_retriever(indexes_manifest: Path, cache_dir: Path | None = None, *, recorder=None) -> TaskRetriever:
    data = json.loads(indexes_manifest.read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "knowledge" in data:
        records = data.get("knowledge", [])
    elif isinstance(data, dict) and isinstance(data.get("resources"), dict):
        records = data["resources"].get("knowledge", [])
    else:
        records = []
    indexes: list[KnowledgeIndex] = []
    embedding = None
    try:
        for item in records:
            path = Path(item["path"] if isinstance(item, dict) else item)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            config = RetrievalConfig.model_validate(item["retrieval"]) if isinstance(item, dict) and "retrieval" in item else None
            mode = config.mode if config else manifest.get("identity", {}).get("retrieval", {}).get("mode", "keyword")
            if mode != "keyword" and embedding is None:
                from lexverse.capabilities.knowledge.embeddings import create_embeddings
                from lexverse.tasks.user import EmbeddingConfig

                emb_cfg = config.embedding if config else EmbeddingConfig.model_validate(manifest.get("embedding_config", {}))
                effective_cache = cache_dir or path.parents[1]
                embedding = create_embeddings(emb_cfg, effective_cache, recorder=recorder)
            indexes.append(KnowledgeIndex(path, embedding, config=config))
        return TaskRetriever(indexes, cache_root=cache_dir or CAPABILITIES_ROOT)
    except BaseException:
        for index in indexes:
            index.close()
        raise


def create_knowledge_server(indexes_manifest: Path, cache_dir: Path | None = None, *, usage_log: Path | None = None):
    """Expose the task's prepared indexes as knowledge MCP tools."""
    from fastmcp import FastMCP

    lock = threading.Lock()

    def record_embedding(kind, texts, tokens, status):
        if usage_log is not None:
            with lock, usage_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"kind": kind, "texts": texts, "tokens": tokens, "status": status}) + "\n")

    @asynccontextmanager
    async def lifespan(server):
        try:
            yield
        finally:
            for index in retriever.indexes.values():
                index.close()

    retriever = _load_retriever(indexes_manifest, cache_dir=cache_dir, recorder=record_embedding)
    default_top_k = next(iter(retriever.indexes.values())).config.top_k if retriever.indexes else 8
    mcp = FastMCP("lexverse-knowledge", lifespan=lifespan)

    @mcp.tool(
        name="search_knowledge",
        description="Search the task's allowed legal knowledge sources and return ranked evidence snippets with citation_ids.",
    )
    def search_knowledge(
        query: Annotated[str, Field(min_length=1, max_length=3000)],
        top_k: Annotated[int, Field(ge=1, le=20)] = default_top_k,
        source_ids: list[str] | None = None,
        filters: KnowledgeFilters | None = None,
    ) -> str:
        results = retriever.search(
            query=query,
            top_k=top_k,
            source_ids=source_ids,
            filters=filters.model_dump(exclude_none=True) if filters is not None else None,
        )
        return json.dumps(results, ensure_ascii=False)

    @mcp.tool(
        name="fetch_knowledge",
        description="Fetch the full text of a legal knowledge chunk by its citation_id (`knowledge:<index_hash>:<chunk_id>`).",
    )
    def fetch_knowledge(
        citation_id: str,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=8000)] = 4000,
    ) -> str:
        result = retriever.fetch(citation_id=citation_id, offset=offset, limit=limit)
        return json.dumps(result, ensure_ascii=False)

    @mcp.resource("knowledge://{source_id}/chunks")
    def list_source_chunks(source_id: str) -> str:
        idx = retriever.indexes.get(source_id)
        if idx is None:
            for cand in retriever.indexes.values():
                if cand.index_hash == source_id or cand.index_hash.startswith(source_id):
                    idx = cand
                    break
        if idx is None:
            raise ValueError(f"Unknown knowledge source: {source_id}")
        rows = idx.db.execute("SELECT id, text, metadata FROM chunks LIMIT 50").fetchall()
        items = [
            {
                "chunk_id": r[0],
                "citation_id": f"knowledge:{idx.index_hash}:{r[0]}",
                "snippet": r[1][:400],
                "metadata": json.loads(r[2]),
            }
            for r in rows
        ]
        return json.dumps(items, ensure_ascii=False)

    data = json.loads(indexes_manifest.read_text(encoding="utf-8"))
    selected = data.get("tools", ["search_knowledge", "fetch_knowledge"]) if isinstance(data, dict) else ["search_knowledge", "fetch_knowledge"]
    for name in ("search_knowledge", "fetch_knowledge"):
        if name not in selected:
            mcp.disable(names={name}, components={"tool"})
    return mcp


def run_stdio_mcp_server(indexes_manifest: Path, cache_dir: Path | None = None, *, usage_log: Path | None = None) -> None:
    create_knowledge_server(indexes_manifest, cache_dir, usage_log=usage_log).run(transport="stdio", show_banner=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LexVerse Legal Knowledge MCP Server")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json or knowledge_indexes.json containing index paths.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional LexVerse capabilities cache directory.",
    )
    parser.add_argument("--usage-log", type=Path, help="Private embedding usage journal for the built-in client.")
    args = parser.parse_args(argv)
    run_stdio_mcp_server(args.manifest, cache_dir=args.cache, usage_log=args.usage_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
