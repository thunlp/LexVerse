"""stdio MCP server exposing the same contract as :class:`CorpusEnv`."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from .env import CorpusEnv
from .errors import error_response

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("MCP server 需要安装 mcp 包") from exc

mcp = FastMCP("lexverse")


@lru_cache(maxsize=1)
def _environment() -> CorpusEnv:
    return CorpusEnv.open(
        os.environ.get("LEXVERSE_DATA_DIR", "data"),
        os.environ.get("LEXVERSE_STATE_DIR", ".lexverse"),
    )


def _call(method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return getattr(_environment(), method)(*args, **kwargs)
    except Exception as exc:
        return error_response(exc)


@mcp.tool()
def list_collections() -> dict[str, Any]:
    """List available legal data collections and capabilities."""
    return _call("list_collections")


@mcp.tool()
def describe_collection(collection: str, provider: str = "local") -> dict[str, Any]:
    """Describe a collection's sources, searchable fields and filters."""
    return _call("describe_collection", collection, provider)


@mcp.tool()
def search_records(collection: str, query: str | None = None,
                   filters: dict[str, Any] | None = None,
                   sort: Literal["relevance", "stable"] = "relevance", limit: int = 10,
                   cursor: str | None = None, provider: str = "local") -> dict[str, Any]:
    """Search records and return only stable ids and minimal key fields."""
    return _call("search_records", collection, query, filters, sort, limit, cursor, provider)


@mcp.tool()
def get_record(record_id: str) -> dict[str, Any]:
    """Fetch one complete source record by its environment id."""
    return _call("get_record", record_id)


@mcp.tool()
def read_record_part(record_id: str, path: str, offset: int = 0,
                     limit: int = 12_000) -> dict[str, Any]:
    """Read a bounded JSON Pointer field slice from a large record."""
    return _call("read_record_part", record_id, path, offset, limit)


@mcp.tool()
def get_asset(
    asset_id: str, mode: Literal["metadata", "text", "binary"] = "metadata"
) -> dict[str, Any]:
    """Read metadata, text, or bounded base64 for a registered template asset."""
    return _call("get_asset", asset_id, mode)


@mcp.tool()
def get_manifest() -> dict[str, Any]:
    """Return the current data snapshot and index manifest."""
    return _call("get_manifest")


def main() -> None:
    # FastMCP owns stdout for JSON-RPC; diagnostics must go to stderr.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
