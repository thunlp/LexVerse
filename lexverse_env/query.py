"""Query, filtering, deterministic ranking and cursor pagination."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from .errors import InvalidArgumentError, InvalidCursorError, UnknownCollectionError
from .index import read_meta
from .utils import (
    decode_cursor,
    encode_cursor,
    make_fts_query,
    normalize_text,
    stable_hash,
    tokenize,
    validate_limit,
)


FILTER_FIELDS_BY_COLLECTION: dict[str, tuple[str, ...]] = {
    "legal_laws": ("source", "category", "year"),
    "legal_cases": (
        "source", "dataset", "category", "case_type", "court", "stage", "year"
    ),
    "legal_qa": ("source", "dataset", "category", "case_type", "year"),
    "legal_concepts": ("source", "dataset", "category", "year"),
    "legal_templates": ("source", "category", "case_type", "year"),
}


@dataclass(frozen=True)
class SearchRequest:
    collection: str
    provider: str = "local"
    query: str | None = None
    filters: dict[str, Any] | None = None
    sort: str = "relevance"
    limit: int = 10
    cursor: str | None = None

    def canonical(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "provider": self.provider,
            "query": normalize_text(self.query or ""),
            "filters": self.filters or {},
            "sort": self.sort,
            "limit": self.limit,
        }


class QueryService:
    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        allowed_collections: set[str] | None = None,
        max_limit: int = 50,
    ):
        self.db = db
        self.meta = read_meta(db)
        self.has_fts = bool(self.meta.get("has_fts", False))
        self.allowed_collections = allowed_collections
        self.max_limit = max_limit

    def _validate(self, request: SearchRequest) -> SearchRequest:
        if not isinstance(request.collection, str) or not request.collection:
            raise InvalidArgumentError("collection 必须是非空字符串")
        if self.allowed_collections is not None and request.collection not in self.allowed_collections:
            raise UnknownCollectionError(f"未知集合: {request.collection}")
        if not isinstance(request.provider, str) or not request.provider:
            raise InvalidArgumentError("provider 必须是非空字符串")
        if request.query is not None and not isinstance(request.query, str):
            raise InvalidArgumentError("query 必须是字符串或 null")
        if request.filters is not None and not isinstance(request.filters, dict):
            raise InvalidArgumentError("filters 必须是对象")
        allowed_filters = set(FILTER_FIELDS_BY_COLLECTION.get(request.collection, ()))
        unknown_filters = sorted((
            key
            for key in (request.filters or {})
            if not isinstance(key, str) or not key or key not in allowed_filters
        ), key=str)
        if unknown_filters:
            names = ", ".join(map(str, unknown_filters))
            raise InvalidArgumentError(f"集合不支持以下过滤字段: {names}")
        for value in (request.filters or {}).values():
            if not (
                value is None
                or isinstance(value, (str, int, float, bool))
            ):
                raise InvalidArgumentError("filter 值必须是字符串、数字、布尔值或 null")
            if isinstance(value, float) and not math.isfinite(value):
                raise InvalidArgumentError("filter 数字必须是有限值")
        if not isinstance(request.sort, str) or request.sort not in {"relevance", "stable"}:
            raise InvalidArgumentError("sort 只能是 relevance 或 stable")
        if request.cursor is not None and (
            not isinstance(request.cursor, str) or not request.cursor
        ):
            raise InvalidArgumentError("cursor 必须是非空字符串或 null")
        limit = validate_limit(request.limit, maximum=self.max_limit)
        return SearchRequest(
            collection=request.collection,
            provider=request.provider,
            query=request.query,
            filters=request.filters or {},
            sort=request.sort,
            limit=limit,
            cursor=request.cursor,
        )

    @staticmethod
    def _filter_matches(fields: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if key not in fields:
                return False
            actual = fields[key]
            if isinstance(actual, list):
                if expected not in actual:
                    return False
            elif str(actual) != str(expected):
                return False
        return True

    @staticmethod
    def _decode_json(value: str, fallback: Any) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _exact_boost(key_fields: dict[str, Any], query: str) -> tuple[int, int]:
        if not query:
            return (0, 0)
        normalized_query = normalize_text(query)
        values = " ".join(normalize_text(value) for value in key_fields.values())
        exact = int(normalized_query in values)
        query_tokens = set(tokenize(normalized_query))
        matched = sum(1 for token in query_tokens if token in values)
        return exact, matched

    def _rows(self, request: SearchRequest) -> list[dict[str, Any]]:
        query = normalize_text(request.query or "")
        filters = request.filters or {}
        rows: list[dict[str, Any]] = []
        if query and self.has_fts:
            fts_query = make_fts_query(query)
            if not fts_query:
                return []
            cursor = self.db.execute(
                """
                SELECT r.id, r.key_fields_json, r.filter_fields_json,
                       r.source, r.source_key, bm25(records_fts) AS rank
                FROM records_fts
                JOIN records AS r ON r.id = records_fts.id
                WHERE r.collection = ? AND r.provider = ?
                  AND records_fts MATCH ?
                """,
                (request.collection, request.provider, fts_query),
            )
            for row in cursor:
                filter_fields = self._decode_json(row["filter_fields_json"], {})
                if not isinstance(filter_fields, dict) or not self._filter_matches(
                    filter_fields, filters
                ):
                    continue
                key_fields = self._decode_json(row["key_fields_json"], {})
                if not isinstance(key_fields, dict):
                    key_fields = {}
                rows.append(
                    {
                        "id": row["id"],
                        "key_fields": key_fields,
                        "source": row["source"],
                        "source_key": row["source_key"],
                        "rank": float(row["rank"] or 0.0),
                    }
                )
        else:
            sql = """
                SELECT id, key_fields_json, filter_fields_json,
                       source, source_key, search_text
                FROM records
                WHERE collection = ? AND provider = ?
            """
            for row in self.db.execute(sql, (request.collection, request.provider)):
                filter_fields = self._decode_json(row["filter_fields_json"], {})
                if not isinstance(filter_fields, dict) or not self._filter_matches(
                    filter_fields, filters
                ):
                    continue
                key_fields = self._decode_json(row["key_fields_json"], {})
                if not isinstance(key_fields, dict):
                    key_fields = {}
                if query:
                    text = normalize_text(row["search_text"])
                    query_tokens = tokenize(query)
                    if query not in text and not any(token in text for token in query_tokens):
                        continue
                rows.append(
                    {
                        "id": row["id"],
                        "key_fields": key_fields,
                        "source": row["source"],
                        "source_key": row["source_key"],
                        "rank": 0.0,
                    }
                )

        if request.sort == "stable" or not query:
            rows.sort(key=lambda row: row["id"])
        else:
            rows.sort(
                key=lambda row: (
                    -self._exact_boost(row["key_fields"], query)[0],
                    -self._exact_boost(row["key_fields"], query)[1],
                    row["rank"],
                    row["id"],
                )
            )
        return rows

    def _stable_page(
        self, request: SearchRequest, offset: int
    ) -> tuple[list[dict[str, Any]], bool]:
        """Stream stable pages without materializing the entire collection."""

        query = normalize_text(request.query or "")
        filters = request.filters or {}
        if query and self.has_fts:
            fts_query = make_fts_query(query)
            if not fts_query:
                return [], False
            rows = self.db.execute(
                """
                SELECT r.id, r.key_fields_json, r.filter_fields_json,
                       r.source, r.source_key
                FROM records_fts
                JOIN records AS r ON r.id = records_fts.id
                WHERE r.collection = ? AND r.provider = ?
                  AND records_fts MATCH ?
                ORDER BY r.id
                """,
                (request.collection, request.provider, fts_query),
            )
        else:
            rows = self.db.execute(
                """
                SELECT id, key_fields_json, filter_fields_json,
                       source, source_key, search_text
                FROM records
                WHERE collection = ? AND provider = ?
                ORDER BY id
                """,
                (request.collection, request.provider),
            )

        matched = 0
        page: list[dict[str, Any]] = []
        for row in rows:
            filter_fields = self._decode_json(row["filter_fields_json"], {})
            if not isinstance(filter_fields, dict) or not self._filter_matches(
                filter_fields, filters
            ):
                continue
            if query and not self.has_fts:
                text = normalize_text(row["search_text"])
                query_tokens = tokenize(query)
                if query not in text and not any(token in text for token in query_tokens):
                    continue
            if matched < offset:
                matched += 1
                continue
            key_fields = self._decode_json(row["key_fields_json"], {})
            page.append(
                {
                    "id": row["id"],
                    "key_fields": key_fields if isinstance(key_fields, dict) else {},
                    "source": row["source"],
                    "source_key": row["source_key"],
                }
            )
            if len(page) > request.limit:
                break
        return page[: request.limit], len(page) > request.limit

    def search(self, request: SearchRequest | None = None, **kwargs: Any) -> dict[str, Any]:
        if request is None:
            request = SearchRequest(**kwargs)
        request = self._validate(request)
        request_hash = stable_hash(request.canonical(), length=32)
        index_generation = str(self.meta.get("index_generation", ""))
        offset = 0
        if request.cursor:
            payload = decode_cursor(request.cursor)
            if (
                payload.get("version") != 1
                or payload.get("request_hash") != request_hash
                or payload.get("index_generation") != index_generation
            ):
                raise InvalidCursorError("cursor 与当前查询或索引版本不匹配")
            offset = payload.get("offset", -1)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise InvalidCursorError("cursor offset 无效")

        if request.sort == "stable" or not normalize_text(request.query or ""):
            page, has_more = self._stable_page(request, offset)
        else:
            rows = self._rows(request)
            page = rows[offset : offset + request.limit]
            has_more = offset + len(page) < len(rows)
        items = []
        for row in page:
            item = {"id": row["id"]}
            if row["key_fields"]:
                item["key_fields"] = row["key_fields"]
            items.append(item)
        next_cursor = None
        next_offset = offset + len(page)
        if has_more:
            next_cursor = encode_cursor(
                {
                    "version": 1,
                    "request_hash": request_hash,
                    "index_generation": index_generation,
                    "offset": next_offset,
                }
            )
        return {"items": items, "next_cursor": next_cursor}
