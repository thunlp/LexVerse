from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import tempfile

from filelock import FileLock
from langchain_core.documents import Document

from lexverse.capabilities.knowledge.loaders import LOADER_VERSION, hash_file, load_documents, scan_source
from lexverse.config import ConfigError
from lexverse.runtime.results import atomic_write_json
from lexverse.tasks.user import KnowledgeFilters, RetrievalConfig


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _index_retrieval(config: RetrievalConfig, embedding_identity: dict) -> dict:
    recipe = config.model_dump()
    # This is the persisted index recipe, not the public YAML schema. Preserve
    # the existing default recipe so changing configuration syntax costs no re-embedding.
    recipe["embedding"] = ("bge_small_zh" if config.mode == "keyword" or config.embedding.uses_default_encoding()
                           else embedding_identity)
    return recipe


def _tokenizer():
    import jieba
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    return tokenizer


def _words(tokenizer, text):
    return [word.strip() for word in tokenizer.cut(text) if word.strip() and any(c.isalnum() for c in word)]


def _append_vectors(index, texts, embedding):
    import faiss
    import numpy as np

    array = np.asarray(embedding.embed_documents(texts), dtype="float32")
    if array.ndim != 2 or array.shape[0] != len(texts) or not array.shape[1] or not np.isfinite(array).all():
        raise ConfigError("Embedding output does not match the chunks")
    if index is None:
        index = faiss.IndexFlatIP(array.shape[1])
    elif index.d != array.shape[1]:
        raise ConfigError("Embedding dimension changed between batches")
    faiss.normalize_L2(array)
    index.add(array)
    return index


def build_index(source: Path, cache: Path, config: RetrievalConfig, embedding=None, *, scan_report=None, progress=None) -> Path:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files, skipped = scan_source(source)
    if scan_report is not None:
        scan_report.update(files=files, skipped=skipped)
    if not files:
        raise ConfigError(f"Knowledge source has no supported files: {source.name}")
    if progress is not None:
        progress(0, len(files), 0)
    source_id = source.name + "_" + _hash(str(source.resolve()))[:8]
    tokenizer = _tokenizer()
    if config.mode != "keyword" and embedding is None:
        raise ConfigError("Vector retrieval requires an embedding model")
    if embedding is None:
        token_length = lambda text: len(_words(tokenizer, text))
        embedding_identity = {"tokenizer": "jieba", "vector": False}
    else:
        token_length = lambda text: len(embedding.tokenizer.encode(text))
        embedding_identity = embedding.identity
        prefix = getattr(embedding, "document_prefix", "")
        overhead = len(embedding.tokenizer.encode(prefix)) if prefix else 2
        if config.chunk_tokens > embedding.identity["max_tokens"] - overhead:
            raise ConfigError("Chunk size exceeds embedding model capacity")
    retrieval_identity = _index_retrieval(config, embedding_identity)
    identity = {
        "format_version": 5, "loader_version": LOADER_VERSION, "source_id": source_id, "files": files,
        "retrieval": retrieval_identity, "embedding": embedding_identity,
        "jieba": importlib.metadata.version("jieba"), "dictionary": hash_file(Path(tokenizer.get_dict_file().name)),
        "splitter": importlib.metadata.version("langchain-text-splitters"),
    }
    index_hash = _hash(identity)
    destination = cache / "knowledge" / index_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            manifest = verify_index(destination)
            if progress is not None:
                progress(len(files), len(files), manifest["chunk_count"])
            return destination
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / "index"
            staged.mkdir()
            db = sqlite3.connect(staged / "documents.sqlite")
            db.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE keywords USING fts5(id UNINDEXED, tokens)")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_tokens, chunk_overlap=config.overlap_tokens,
                length_function=token_length, add_start_index=True,
                separators=["\n\n", "\n", "。", "；", " ", ""],
            )
            vector_index, ids = None, []
            completed_files = 0

            def file_completed(relative):
                nonlocal completed_files
                completed_files += 1
                if progress is not None:
                    progress(completed_files, len(files), len(ids))

            try:
                batch = []
                for document in load_documents(source, files, source_id, cache=cache, file_completed=file_completed):
                    previous_start = -1
                    previous_text = ""
                    for ordinal, chunk in enumerate(splitter.split_documents([document])):
                        # Token overlap is not a reliable character offset in repeated text.
                        extends_previous = previous_text and len(chunk.page_content) > len(previous_text) and chunk.page_content.startswith(previous_text)
                        start = document.page_content.find(chunk.page_content, previous_start if extends_previous else previous_start + 1)
                        if start < 0:
                            raise ConfigError(f"Cannot locate chunk {ordinal} in document {document.metadata['document_id']}")
                        chunk.metadata.update(start_index=start, end_index=start + len(chunk.page_content))
                        previous_start = start
                        previous_text = chunk.page_content
                        chunk_id = _hash({"document": document.metadata["document_id"],
                                          "content": document.metadata["content_hash"],
                                          "start": chunk.metadata["start_index"], "text": chunk.page_content,
                                          "ordinal": ordinal,
                                          "config": retrieval_identity})
                        chunk.metadata.update({"chunk_id": chunk_id, "chunk_ordinal": ordinal, "index_hash": index_hash,
                                               "citation_id": f"knowledge:{index_hash}:{chunk_id}",
                                               "text_hash": hashlib.sha256(chunk.page_content.encode()).hexdigest()})
                        db.execute("INSERT INTO chunks VALUES (?, ?, ?)",
                                   (chunk_id, chunk.page_content, json.dumps(chunk.metadata, ensure_ascii=False)))
                        db.execute("INSERT INTO keywords VALUES (?, ?)",
                                   (chunk_id, " ".join(_words(tokenizer, chunk.page_content))))
                        ids.append(chunk_id)
                        if progress is not None and len(ids) % 64 == 0:
                            progress(completed_files, len(files), len(ids))
                        if embedding is not None and config.mode != "keyword":
                            batch.append(chunk.page_content)
                            if len(batch) == 64:
                                vector_index = _append_vectors(vector_index, batch, embedding)
                                batch.clear()
                if batch:
                    vector_index = _append_vectors(vector_index, batch, embedding)
                if not ids:
                    raise ConfigError("Knowledge parsing produced no chunks")
                db.commit()
            finally:
                db.close()
            if vector_index is not None:
                import faiss
                if vector_index.ntotal != len(ids):
                    raise ConfigError("Vector index and chunks differ")
                faiss.write_index(vector_index, str(staged / "vectors.faiss"))
                atomic_write_json(staged / "vector_ids.json", ids)
            manifest = {"index_hash": index_hash, "identity": identity, "skipped": skipped,
                        "embedding_config": config.embedding.model_dump(mode="json"),
                        "chunk_count": len(ids), "source_name": source.name,
                        "files": {p.name: hash_file(p) for p in staged.iterdir() if p.is_file()}}
            atomic_write_json(staged / "manifest.json", manifest)
            staged.rename(destination)
    return destination


def verify_index(path: Path) -> dict:
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["index_hash"] != path.name or _hash(manifest["identity"]) != path.name:
            raise ValueError("Index identity mismatch")
        for name, expected in manifest["files"].items():
            if Path(name).name != name or hash_file(path / name) != expected:
                raise ValueError("Index content mismatch")
        return manifest
    except (OSError, ValueError, KeyError) as exc:
        raise ConfigError("Invalid or damaged knowledge index") from exc


def preparation_report(index, cache: Path, *, skipped=None) -> dict:
    root = cache / "knowledge_preparation"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (index.source_id + ".json")
    current = index.manifest["identity"]["files"]
    with FileLock(str(path) + ".lock"):
        previous = json.loads(path.read_text()) if path.exists() else {"files": {}, "index_hash": None}
        old = previous["files"]
        changes = {
            "added": sorted(current.keys() - old.keys()),
            "modified": sorted(name for name in current.keys() & old.keys() if current[name] != old[name]),
            "deleted": sorted(old.keys() - current.keys()),
        }
        report = {"source_id": index.source_id, "source_name": index.manifest["source_name"],
                  "index_hash": index.index_hash, "previous_index_hash": previous["index_hash"],
                  "files": current, "skipped": index.manifest["skipped"] if skipped is None else skipped,
                  "chunk_count": index.manifest["chunk_count"], "changes": changes,
                  "change_counts": {kind: len(names) for kind, names in changes.items()}}
        atomic_write_json(path, {"files": current, "index_hash": index.index_hash})
    return report


class KnowledgeIndex:
    def __init__(self, path: Path, embedding=None):
        self.manifest = verify_index(path)
        self.index_hash = self.manifest["index_hash"]
        self.source_id = self.manifest["identity"]["source_id"]
        recipe = dict(self.manifest["identity"]["retrieval"])
        # Encoding identity is checked separately below; retrieval only needs
        # search/chunk settings from the persisted recipe.
        recipe.pop("embedding", None)
        self.config = RetrievalConfig.model_validate(recipe)
        self.db = sqlite3.connect((path / "documents.sqlite").resolve().as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.tokenizer = _tokenizer()
        self.vector_store = None
        if self.config.mode != "keyword":
            import faiss
            from langchain_community.docstore.base import Docstore
            from langchain_community.vectorstores import FAISS
            from langchain_community.vectorstores.utils import DistanceStrategy

            if embedding is None or embedding.identity != self.manifest["identity"]["embedding"]:
                self.db.close()
                raise ConfigError("Embedding identity does not match the knowledge index")
            owner = self
            class SQLiteDocstore(Docstore):
                def search(self, search: str):
                    return owner.document(search)
            ids = json.loads((path / "vector_ids.json").read_text())
            vector = faiss.read_index(str(path / "vectors.faiss"))
            if vector.ntotal != len(ids) or len(ids) != self.manifest["chunk_count"]:
                self.db.close()
                raise ConfigError("Vector index and document mapping differ")
            self.vector_store = FAISS(embedding, vector, SQLiteDocstore(), dict(enumerate(ids)),
                                      distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

    def close(self):
        self.db.close()

    def document(self, chunk_id: str) -> Document:
        row = self.db.execute("SELECT text, metadata FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown knowledge chunk")
        return Document(page_content=row[0], metadata=json.loads(row[1]))

    def search(self, query: str, filters: dict | None = None) -> list[tuple[Document, float]]:
        ranks = []
        if self.config.mode != "vector":
            terms = list(dict.fromkeys(_words(self.tokenizer, query)))[:100]
            if terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                conditions = "".join(" AND json_extract(chunks.metadata, ?) = ?" for _ in (filters or {}))
                arguments = [value for key, item in (filters or {}).items() for value in ("$." + key, item)]
                rows = self.db.execute(
                    "SELECT keywords.id FROM keywords JOIN chunks ON chunks.id = keywords.id "
                    "WHERE keywords MATCH ?" + conditions + " ORDER BY bm25(keywords) LIMIT ?",
                    [expression, *arguments, self.config.keyword_candidates],
                ).fetchall()
                ranks.append([row[0] for row in rows])
        if self.vector_store is not None:
            import faiss
            # OpenMP settings are thread-local; tools can search from worker threads.
            faiss.omp_set_num_threads(1)
            hits = self.vector_store.similarity_search(
                query, k=self.config.vector_candidates, filter=filters,
                fetch_k=self.manifest["chunk_count"] if filters else self.config.vector_candidates,
            )
            ranks.append([doc.metadata["chunk_id"] for doc in hits])
        scores = {}
        for ranking in ranks:
            for rank, chunk_id in enumerate(ranking, 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
        return [(self.document(chunk_id), score) for chunk_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


class TaskRetriever:
    def __init__(self, indexes: list[KnowledgeIndex]):
        self.indexes = {index.source_id: index for index in indexes}

    def search(self, query: str, top_k: int = 8, source_ids: list[str] | None = None,
               filters: dict | None = None) -> list[dict]:
        if not query.strip() or len(query) > 3000 or not 1 <= top_k <= 20:
            raise ValueError("Query or top_k exceeds allowed limits")
        filters = KnowledgeFilters.model_validate(filters or {}).model_dump(exclude_none=True)
        selected = list(self.indexes) if source_ids is None else list(dict.fromkeys(source_ids))
        if set(selected) - self.indexes.keys():
            raise ValueError("Knowledge source is not allowed for this task")
        hits = []
        for source in selected:
            for rank, (document, _) in enumerate(self.indexes[source].search(query, filters), 1):
                hits.append((1 / (60 + rank), document))
        results, counts = [], {}
        for score, doc in sorted(hits, key=lambda pair: (-pair[0], pair[1].metadata["chunk_id"])):
            parent = doc.metadata["document_id"]
            if counts.get(parent, 0) >= 2:
                continue
            counts[parent] = counts.get(parent, 0) + 1
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
