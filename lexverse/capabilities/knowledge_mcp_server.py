from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lexverse.capabilities.knowledge.index import KnowledgeIndex, TaskRetriever
from lexverse.tasks.user import RetrievalConfig


def _load_retriever(indexes_manifest: Path, cache_dir: Path | None = None) -> TaskRetriever:
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
    for item in records:
        path = Path(item["path"] if isinstance(item, dict) else item)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        mode = manifest.get("identity", {}).get("retrieval", {}).get("mode", "keyword")
        if mode != "keyword" and embedding is None:
            from lexverse.capabilities.knowledge.embeddings import create_embeddings
            from lexverse.tasks.user import EmbeddingConfig

            emb_cfg = EmbeddingConfig.model_validate(manifest.get("embedding_config", {}))
            effective_cache = cache_dir or path.parents[1]
            embedding = create_embeddings(emb_cfg, effective_cache)
        config = RetrievalConfig.model_validate(item["retrieval"]) if isinstance(item, dict) and "retrieval" in item else None
        indexes.append(KnowledgeIndex(path, embedding, config=config))
    return TaskRetriever(indexes)


def run_stdio_mcp_server(indexes_manifest: Path, cache_dir: Path | None = None) -> None:
    """
    Start a standard MCP stdio server exposing `search_knowledge` and `fetch_knowledge`
    over the task's prepared KnowledgeIndex instances.
    """
    from fastmcp import FastMCP

    retriever = _load_retriever(indexes_manifest, cache_dir=cache_dir)
    default_top_k = next(iter(retriever.indexes.values())).config.top_k if retriever.indexes else 8
    mcp = FastMCP("lexverse-knowledge")

    @mcp.tool(
        name="search_knowledge",
        description="Search the task's allowed legal knowledge sources and return ranked evidence snippets with citation_ids.",
    )
    def search_knowledge(
        query: str,
        top_k: int = default_top_k,
        source_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> str:
        results = retriever.search(
            query=query,
            top_k=top_k,
            source_ids=source_ids,
            filters=filters,
        )
        return json.dumps(results, ensure_ascii=False)

    @mcp.tool(
        name="fetch_knowledge",
        description="Fetch the full text of a legal knowledge chunk by its citation_id (`knowledge:<index_hash>:<chunk_id>`).",
    )
    def fetch_knowledge(
        citation_id: str,
        offset: int = 0,
        limit: int = 4000,
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

    mcp.run(transport="stdio")


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
    args = parser.parse_args(argv)
    run_stdio_mcp_server(args.manifest, cache_dir=args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
