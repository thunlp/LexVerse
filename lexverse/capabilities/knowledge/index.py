from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from filelock import FileLock

from lexverse.capabilities.knowledge.loaders import (
    Document, KINDS, LOADER_VERSION, hash_file, load_documents, scan_source, source_kind,
)
from lexverse.capabilities.knowledge.keyword import create_tokenizer, tokenize
from lexverse.config import ConfigError
from lexverse.runtime.results import atomic_write_json
from lexverse.tasks.user import RetrievalConfig


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def is_index_directory(path: Path) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", path.name) is not None and (path / "manifest.json").is_file()


def _build_recipe(recipe: dict) -> dict:
    return {key: value for key, value in recipe.items()
            if key not in {"top_k", "keyword_candidates", "vector_candidates"}}


def _index_retrieval(config: RetrievalConfig, embedding_identity: dict) -> dict:
    recipe = _build_recipe(config.model_dump())
    # Keep the default embedding label compatible with published indexes;
    # query settings are excluded from the build recipe.
    recipe["embedding"] = ("bge_small_zh" if config.mode == "keyword" or config.embedding.uses_default_encoding()
                           else embedding_identity)
    return recipe


class _NumpyFlatIPIndex:
    """Lightweight fallback for faiss.IndexFlatIP when faiss-cpu is not installed."""

    def __init__(self, d: int, vectors=None):
        import numpy as np

        self.d = d
        self._vectors = (
            np.empty((0, d), dtype="float32")
            if vectors is None
            else np.asarray(vectors, dtype="float32")
        )

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, array) -> None:
        import numpy as np

        arr = np.asarray(array, dtype="float32")
        self._vectors = np.vstack([self._vectors, arr]) if self.ntotal else arr.copy()

    def search(self, query_array, k: int):
        import numpy as np

        q = np.asarray(query_array, dtype="float32")
        if self.ntotal == 0 or k <= 0:
            return np.empty((q.shape[0], 0), dtype="float32"), np.empty((q.shape[0], 0), dtype="int64")
        scores = q @ self._vectors.T
        top_k = min(k, self.ntotal)
        idx = np.argsort(-scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, idx, axis=1)
        return top_scores, idx.astype("int64")


def _normalize_l2_inplace(array) -> None:
    import numpy as np

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    array /= norms


def _write_vector_index(index, path: Path) -> None:
    try:
        import faiss

        if isinstance(index, _NumpyFlatIPIndex):
            idx = faiss.IndexFlatIP(index.d)
            if index.ntotal:
                idx.add(index._vectors)
            faiss.write_index(idx, str(path))
        else:
            faiss.write_index(index, str(path))
    except ImportError:
        import numpy as np

        with path.open("wb") as f:
            np.save(f, index._vectors, allow_pickle=False)


def _read_vector_index(path: Path):
    try:
        import faiss

        return faiss.read_index(str(path))
    except ImportError:
        import numpy as np

        with path.open("rb") as f:
            arr = np.load(f, allow_pickle=False)
        return _NumpyFlatIPIndex(int(arr.shape[1]), arr)


def _append_vectors(index, texts, embedding):
    import numpy as np

    array = np.asarray(embedding.embed_documents(texts), dtype="float32")
    if array.ndim != 2 or array.shape[0] != len(texts) or not array.shape[1] or not np.isfinite(array).all():
        raise ConfigError("Embedding output does not match the chunks")
    try:
        import faiss

        if index is None:
            index = faiss.IndexFlatIP(array.shape[1])
        elif index.d != array.shape[1]:
            raise ConfigError("Embedding dimension changed between batches")
        faiss.normalize_L2(array)
        index.add(array)
        return index
    except ImportError:
        if index is None:
            index = _NumpyFlatIPIndex(int(array.shape[1]))
        elif index.d != array.shape[1]:
            raise ConfigError("Embedding dimension changed between batches")
        _normalize_l2_inplace(array)
        index.add(array)
        return index


def _build_text_splitter(chunk_size: int, chunk_overlap: int, length_function):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        )
    except ImportError:
        class _FallbackSplitter:
            def split_documents(self, docs):
                out = []
                for doc in docs:
                    text = doc.page_content
                    if length_function(text) <= chunk_size:
                        out.append(Document(page_content=text, metadata=dict(doc.metadata)))
                        continue
                    pos = 0
                    n = len(text)
                    while pos < n:
                        end = n
                        if length_function(text[pos:end]) <= chunk_size:
                            sub = text[pos:end]
                            if sub.strip():
                                out.append(Document(page_content=sub, metadata=dict(doc.metadata)))
                            break
                        est = pos + max(1, int((n - pos) * (chunk_size / max(length_function(text[pos:]), 1))))
                        end = min(n, est)
                        while end > pos + 1 and length_function(text[pos:end]) > chunk_size:
                            end = max(pos + 1, end - 64)
                        for sep in ("\n\n", "\n", "。", ". ", " "):
                            b = text.rfind(sep, pos + 1, end)
                            if b > pos + (end - pos) // 2:
                                end = b + len(sep)
                                break
                        sub = text[pos:end]
                        if sub.strip():
                            out.append(Document(page_content=sub, metadata=dict(doc.metadata)))
                        overlap_chars = max(0, int((end - pos) * (chunk_overlap / max(chunk_size, 1))))
                        pos = max(pos + 1, end - overlap_chars) if end < n else n
                return out

        return _FallbackSplitter()


def build_index(source: Path, cache: Path, config: RetrievalConfig, embedding=None, *, scan_report=None, progress=None) -> Path:

    if is_index_directory(source):
        manifest = verify_index(source)
        identity = manifest["identity"]
        if _build_recipe(identity["retrieval"]) != _index_retrieval(config, identity["embedding"]):
            raise ConfigError("Retrieval configuration does not match the prebuilt knowledge index")
        if config.mode != "keyword" and (embedding is None or embedding.identity != identity["embedding"]):
            raise ConfigError("Embedding identity does not match the knowledge index")
        if scan_report is not None:
            scan_report.update(files=identity["files"], skipped=manifest["skipped"])
        if progress is not None:
            progress(len(identity["files"]), len(identity["files"]), manifest["chunk_count"])
        return source

    files, skipped = scan_source(source)
    if scan_report is not None:
        scan_report.update(files=files, skipped=skipped)
    if not files:
        raise ConfigError(f"Knowledge source has no supported files: {source.name}")
    if progress is not None:
        progress(0, len(files), 0)
    # Category-relative names survive relocation; arbitrary user sources use
    # their directory name. Keep content hashes separate for change reports.
    logical_name = source.name
    for parent in (source, *source.parents):
        if parent.name in KINDS:
            logical_name = source.relative_to(parent.parent).as_posix()
            break
    source_id = source.name + "_" + _hash(logical_name)[:8]
    tokenizer = create_tokenizer()
    if config.mode != "keyword" and embedding is None:
        raise ConfigError("Vector retrieval requires an embedding model")
    if embedding is None:
        token_length = lambda text: len(tokenize(tokenizer, text))
        embedding_identity = {"tokenizer": "jieba", "vector": False}
    else:
        token_length = lambda text: len(embedding.tokenizer.encode(text))
        embedding_identity = embedding.identity
        prefix = getattr(embedding, "document_prefix", "")
        overhead = len(embedding.tokenizer.encode(prefix)) if prefix else 2
        if config.chunk_tokens > embedding.identity["max_tokens"] - overhead:
            raise ConfigError("Chunk size exceeds embedding model capacity")
    retrieval_identity = _index_retrieval(config, embedding_identity)
    try:
        splitter_version = importlib.metadata.version("langchain-text-splitters")
    except importlib.metadata.PackageNotFoundError:
        splitter_version = "fallback-1.0"
    identity = {
        "format_version": 6, "loader_version": LOADER_VERSION, "source_id": source_id, "files": files,
        "kinds": {relative: source_kind((source / relative).parent) for relative in files},
        "retrieval": retrieval_identity, "embedding": embedding_identity,
        "jieba": importlib.metadata.version("jieba"), "dictionary": hash_file(Path(tokenizer.get_dict_file().name)),
        "splitter": splitter_version,
    }
    index_hash = _hash(identity)
    destination = cache / "indexes" / index_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            manifest = verify_index(destination)
            if progress is not None:
                progress(len(files), len(files), manifest["chunk_count"])
            return destination
        # Reuse older, verified indexes whose identity included query settings.
        # Keep their manifests and citation IDs unchanged.
        for candidate in sorted(destination.parent.glob("*/manifest.json")):
            try:
                previous = json.loads(candidate.read_text())["identity"]
                comparable = {**previous, "retrieval": _build_recipe(previous["retrieval"])}
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if comparable == identity:
                manifest = verify_index(candidate.parent)
                if progress is not None:
                    progress(len(files), len(files), manifest["chunk_count"])
                return candidate.parent
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / "index"
            staged.mkdir()
            db = sqlite3.connect(staged / "documents.sqlite")
            db.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE keywords USING fts5(id UNINDEXED, tokens)")
            splitter = _build_text_splitter(
                config.chunk_tokens,
                config.overlap_tokens,
                token_length,
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
                                   (chunk_id, " ".join(tokenize(tokenizer, chunk.page_content))))
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
                if vector_index.ntotal != len(ids):
                    raise ConfigError("Vector index and chunks differ")
                _write_vector_index(vector_index, staged / "vectors.faiss")
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
    root = cache / "cache" / "knowledge_preparation"
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
    def __init__(self, path: Path, embedding=None, *, config: RetrievalConfig | None = None):
        self.path = path.resolve()
        self.manifest = verify_index(path)
        self.index_hash = self.manifest["index_hash"]
        self.source_id = self.manifest["identity"]["source_id"]
        recipe = dict(self.manifest["identity"]["retrieval"])
        # Encoding identity is checked separately below; retrieval only needs
        # search/chunk settings from the persisted recipe.
        recipe.pop("embedding", None)
        if config is not None:
            identity = self.manifest["identity"]
            if _build_recipe(identity["retrieval"]) != _index_retrieval(config, identity["embedding"]):
                raise ConfigError("Retrieval configuration does not match the prebuilt knowledge index")
        self.config = config.model_copy(deep=True) if config is not None else RetrievalConfig.model_validate(recipe)
        self.db = sqlite3.connect((path / "documents.sqlite").resolve().as_uri() + "?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.faiss_index = None
        self.vector_ids: list[str] = []
        self.embedding = embedding
        if self.config.mode != "keyword":
            if embedding is None or embedding.identity != self.manifest["identity"]["embedding"]:
                self.db.close()
                raise ConfigError("Embedding identity does not match the knowledge index")
            ids = json.loads((path / "vector_ids.json").read_text())
            vector = _read_vector_index(path / "vectors.faiss")
            if vector.ntotal != len(ids) or len(ids) != self.manifest["chunk_count"]:
                self.db.close()
                raise ConfigError("Vector index and document mapping differ")
            self.faiss_index = vector
            self.vector_ids = ids

    def close(self):
        self.db.close()

    def document(self, chunk_id: str) -> Document:
        row = self.db.execute("SELECT text, metadata FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown knowledge chunk")
        return Document(page_content=row[0], metadata=json.loads(row[1]))

    def vector_candidates(self, query_vector, filters: dict | None = None, *, limit: int | None = None) -> list[tuple[Document, float]]:
        import numpy as np

        try:
            import faiss
            faiss.omp_set_num_threads(1)
        except ImportError:
            pass
        limit = self.config.vector_candidates if limit is None else limit
        q_vec = np.asarray([query_vector], dtype="float32")
        _normalize_l2_inplace(q_vec)
        filters = dict(filters or {})
        if filters.get("kind") and set(self.manifest["identity"]["kinds"].values()) == {filters["kind"]}:
            filters.pop("kind")
        if self.faiss_index is not None:
            fetch_k = self.manifest["chunk_count"] if filters else min(limit, len(self.vector_ids))
            scores, indices = self.faiss_index.search(q_vec, fetch_k)
            hits = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.vector_ids):
                    continue
                doc = self.document(self.vector_ids[idx])
                if any(doc.metadata.get(k) != v for k, v in filters.items()):
                    continue
                hits.append((doc, float(score)))
                if len(hits) >= limit:
                    break
            return hits
        raise ConfigError("Vector retrieval requires a loaded vector index")
