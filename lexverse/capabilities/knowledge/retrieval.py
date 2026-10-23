from pathlib import Path
import hashlib
import re

from lexverse.capabilities.knowledge.index import KnowledgeIndex
from lexverse.capabilities.knowledge.keyword import KeywordSearch
from lexverse.config import ConfigError
from lexverse.runtime.paths import CAPABILITIES_ROOT
from lexverse.tasks.user import KnowledgeFilters


def _fuse(rankings):
    scores = {}
    for ranking in rankings:
        for rank, (index_hash, chunk_id, _) in enumerate(ranking, 1):
            key = (index_hash, chunk_id)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    return [(index_hash, chunk_id, score) for (index_hash, chunk_id), score
            in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _deduplicate(rankings, by_hash, documents):
    contents, representatives = {}, {}
    for ranking in rankings:
        for rank, (index_hash, chunk_id, _) in enumerate(ranking, 1):
            key = (index_hash, chunk_id)
            if key not in documents:
                documents[key] = by_hash[index_hash].document(chunk_id)
            content = hashlib.sha256(" ".join(documents[key].page_content.split()).encode()).hexdigest()
            contents[key] = content
            choice = (rank, index_hash, chunk_id)
            if content not in representatives or choice < representatives[content]:
                representatives[content] = choice
    channels = []
    for ranking in rankings:
        seen, unique = set(), []
        for index_hash, chunk_id, score in ranking:
            content = contents[index_hash, chunk_id]
            if content in seen:
                continue
            seen.add(content)
            _, index_hash, chunk_id = representatives[content]
            unique.append((index_hash, chunk_id, score))
        channels.append(unique)
    return channels


def _exact_article_hits(query, indexes, filters):
    articles = set(re.findall(r"第[零〇一二三四五六七八九十百千万两0-9]+条(?:之[零〇一二三四五六七八九十百千万两0-9]+)?", query))
    if len(articles) != 1:
        return []
    # Resolve only names present in the selected sources; do not guess law aliases.
    titles = set()
    for index in indexes:
        titles.update(row[0] for row in index.db.execute(
            "SELECT DISTINCT json_extract(metadata, '$.title') FROM chunks") if row[0])
    mentions = {title: title.removeprefix("中华人民共和国") for title in titles}
    matched = {title: name for title, name in mentions.items() if name and name in query}
    matched = {title for title, name in matched.items()
               if not any(name != other and name in other for other in matched.values())}
    if len(matched) != 1:
        return []
    conditions = "".join(f" AND json_extract(metadata, '$.{key}')=?" for key in filters)
    hits = []
    for index in indexes:
        rows = index.db.execute(
            "SELECT id FROM chunks WHERE json_extract(metadata, '$.title')=? "
            "AND json_extract(metadata, '$.article')=?" + conditions + " ORDER BY id",
            [next(iter(matched)), next(iter(articles)), *filters.values()],
        )
        hits.extend((index.index_hash, row[0], 0.0) for row in rows)
    return hits


class TaskRetriever:
    def __init__(self, indexes: list[KnowledgeIndex], *, cache_root: Path = CAPABILITIES_ROOT):
        self.keyword_cache = cache_root / "cache" / "keyword_search"
        self.keyword_search = None
        self.indexes = {}
        for index in indexes:
            previous = self.indexes.get(index.source_id)
            if previous is not None and previous.index_hash != index.index_hash:
                raise ConfigError("Knowledge sources share a logical name; use distinct source directory names")
            self.indexes[index.source_id] = index

    def search(self, query: str, top_k: int = 8, source_ids: list[str] | None = None,
               filters: dict | None = None) -> list[dict]:
        if not query.strip() or len(query) > 3000 or not 1 <= top_k <= 20:
            raise ValueError("Query or top_k exceeds allowed limits")
        filters = KnowledgeFilters.model_validate(filters or {}).model_dump(exclude_none=True)
        if source_ids is None:
            selected = list(self.indexes)
        else:
            resolved = []
            for sid in dict.fromkeys(source_ids):
                if sid in self.indexes:
                    resolved.append(sid)
                else:
                    matched = next(
                        (k for k, cand in self.indexes.items() if cand.index_hash == sid or cand.index_hash.startswith(sid)),
                        None,
                    )
                    resolved.append(matched or sid)
            selected = list(dict.fromkeys(resolved))
        if set(selected) - self.indexes.keys():
            raise ValueError("Knowledge source is not allowed for this task")
        selected_indexes = [self.indexes[source] for source in selected]
        keyword_indexes = [index for index in selected_indexes if index.config.mode != "vector"]
        vector_indexes = [index for index in selected_indexes if index.config.mode != "keyword"
                          and ("kind" not in filters or filters["kind"] in index.manifest["identity"]["kinds"].values())]
        if any(index.manifest["identity"]["embedding"] != vector_indexes[0].manifest["identity"]["embedding"]
               for index in vector_indexes[1:]):
            raise ConfigError("Cross-index vector ranking requires the same embedding identity")
        rankings = []
        if keyword_indexes:
            if self.keyword_search is None:
                collection = [index for index in self.indexes.values() if index.config.mode != "vector"]
                self.keyword_search = KeywordSearch(collection, self.keyword_cache)
            rankings.append(self.keyword_search.search(
                query, [index.index_hash for index in keyword_indexes], filters,
                max(index.config.keyword_candidates for index in keyword_indexes),
            ))
        if vector_indexes:
            query_vector = vector_indexes[0].embedding.embed_query(query)
            candidates = [(index.index_hash, doc.metadata["chunk_id"], float(score))
                          for index in vector_indexes for doc, score in index.vector_candidates(
                              query_vector, filters, limit=index.config.vector_candidates)]
            candidates.sort(key=lambda row: (-row[2], row[0], row[1]))
            rankings.append(candidates[:max(index.config.vector_candidates for index in vector_indexes)])
        by_hash = {index.index_hash: index for index in selected_indexes}
        documents = {}
        hits = (_fuse(_deduplicate(rankings, by_hash, documents)) if len(rankings) > 1
                else (rankings[0] if rankings else []))
        exact = _exact_article_hits(query, selected_indexes, filters)
        if exact:
            scores = {(index_hash, chunk_id): score for index_hash, chunk_id, score in hits}
            keys = {(index_hash, chunk_id) for index_hash, chunk_id, _ in exact}
            hits = [(index_hash, chunk_id, scores.get((index_hash, chunk_id), score))
                    for index_hash, chunk_id, score in exact] + [row for row in hits if row[:2] not in keys]
        results, counts, seen = [], {}, set()
        for index_hash, chunk_id, score in hits:
            doc = documents.get((index_hash, chunk_id)) or by_hash[index_hash].document(chunk_id)
            parent = (index_hash, doc.metadata["document_id"])
            content = hashlib.sha256(" ".join(doc.page_content.split()).encode()).hexdigest()
            if content in seen or counts.get(parent, 0) >= 2:
                continue
            counts[parent] = counts.get(parent, 0) + 1
            seen.add(content)
            results.append({**doc.metadata, "snippet": doc.page_content[:1000], "rank_score": score})
            if len(results) == top_k:
                break
        return results

    def fetch(self, citation_id: str, offset: int = 0, limit: int = 4000) -> dict:
        parts = citation_id.split(":")
        if len(parts) != 3 or parts[0] != "knowledge" or offset < 0 or not 1 <= limit <= 8000:
            raise ValueError("Invalid knowledge citation or page bounds")
        index = next((index for index in self.indexes.values() if index.index_hash == parts[1]), None)
        if index is None:
            raise ValueError("Knowledge citation is not allowed for this task")
        doc = index.document(parts[2])
        return {**doc.metadata, "text": doc.page_content[offset:offset + limit], "offset": offset,
                "next_offset": offset + limit if offset + limit < len(doc.page_content) else None}
