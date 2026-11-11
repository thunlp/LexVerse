from pathlib import Path
import hashlib
import json
import math
import sqlite3
import struct
import sys
import threading

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:  # type: ignore[no-redef]
        pass
from lexverse.config import ConfigError
from lexverse.models.task.config import EmbeddingConfig


def _download_embedding_model(model: str, revision: str | None, cache: Path, files: list[str]) -> str:
    from huggingface_hub import constants, snapshot_download
    from requests import RequestException

    print(f"Downloading embedding model {model}@{revision or 'main'} to {cache} "
          "(HTTP; metadata timeout 15s, read timeout 90s). The first download may take several minutes.",
          file=sys.stderr, flush=True)
    # Hub reads these constants during transfers; restore them for other downloads.
    previous_xet = constants.HF_HUB_DISABLE_XET
    previous_timeout = constants.HF_HUB_DOWNLOAD_TIMEOUT
    try:
        constants.HF_HUB_DISABLE_XET = True
        constants.HF_HUB_DOWNLOAD_TIMEOUT = 90
        snapshot = snapshot_download(model, revision=revision, cache_dir=str(cache),
                                     allow_patterns=files, max_workers=1, etag_timeout=15)
    except (OSError, RequestException) as exc:
        raise ConfigError(f"Cannot download embedding model {model}@{revision or 'main'} to {cache}. "
                          "Check network access to Hugging Face and available disk space, then rerun the command. "
                          "Cached files are retained so the download can resume.") from exc
    finally:
        constants.HF_HUB_DISABLE_XET = previous_xet
        constants.HF_HUB_DOWNLOAD_TIMEOUT = previous_timeout
    print(f"Embedding model download complete: {snapshot}", file=sys.stderr, flush=True)
    return snapshot


class LocalEmbeddings(Embeddings):
    def __init__(self, cache: Path, *, config: EmbeddingConfig | None = None, recorder=None):
        config = config or EmbeddingConfig()
        self._encode_lock = threading.RLock()
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from sentence_transformers import SentenceTransformer
        from lexverse.capabilities.knowledge.loaders import hash_file

        snapshot = Path(config.model).expanduser()
        revision = None
        if not snapshot.is_dir():
            try:
                snapshot = Path(snapshot_download(config.model, revision=config.revision,
                                cache_dir=str(cache / "cache" / "models"), local_files_only=True))
            except LocalEntryNotFoundError:
                snapshot = Path(_download_embedding_model(config.model, config.revision, cache / "cache" / "models",
                                ["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"]))
            if not (snapshot / "modules.json").is_file():
                snapshot = Path(_download_embedding_model(config.model, config.revision, cache / "cache" / "models",
                                ["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"]))
            revision = snapshot.name
        if not (snapshot / "modules.json").is_file():
            raise ConfigError("Custom embedding model must be a Sentence Transformers model (modules.json required)")
        self.model = SentenceTransformer(str(snapshot), device=config.device,
                                        local_files_only=True, trust_remote_code=False)
        capacity = self.model.max_seq_length
        if config.max_tokens is not None and config.max_tokens > capacity:
            raise ConfigError(f"Embedding max_tokens exceeds model capacity {capacity}")
        self.max_tokens = config.max_tokens or capacity
        self.model.max_seq_length = self.max_tokens
        self.tokenizer = self.model.tokenizer
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        self.embedding_dimension = dim_fn()
        if not self.embedding_dimension:
            raise ConfigError("Embedding model must have a fixed output dimension")
        self.batch_size = config.batch_size
        self.query_prefix = config.query_prefix
        self.document_prefix = config.document_prefix
        self.recorder = recorder
        files = {path.relative_to(snapshot).as_posix(): hash_file(path)
                 for path in sorted(snapshot.rglob("*"))
                 if path.is_file() and not any(part.startswith(".") for part in path.relative_to(snapshot).parts)
                 and path.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}}
        self.identity = {"model": "sentence_transformers", "revision": revision, "files": files,
                         "dimension": self.embedding_dimension, "normalize": True,
                         "query_prefix": self.query_prefix, "document_prefix": self.document_prefix,
                         "max_tokens": self.max_tokens}
        identity_hash = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.vector_cache = cache / "cache" / "embedding_cache" / (identity_hash + ".sqlite")


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if getattr(self, "vector_cache", None) is not None:
            return self._cached_documents(texts)
        return self._encode(texts, "documents")

    def _cached_documents(self, texts):
        self.vector_cache.parent.mkdir(parents=True, exist_ok=True)
        db = None
        try:
            db = sqlite3.connect(self.vector_cache, timeout=60)
            db.execute("CREATE TABLE IF NOT EXISTS vectors (text_hash TEXT PRIMARY KEY, vector BLOB, checksum TEXT)")
            keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
            vectors, missing = {}, {}
            for key, text in zip(keys, texts):
                row = db.execute("SELECT vector, checksum FROM vectors WHERE text_hash = ?", (key,)).fetchone()
                if row is None:
                    missing[key] = text
                else:
                    blob, checksum = row
                    if len(blob) != 4 * self.embedding_dimension or hashlib.sha256(blob).hexdigest() != checksum:
                        raise ValueError("Embedding cache is damaged")
                    vectors[key] = list(struct.unpack("<" + "f" * self.embedding_dimension, blob))
                    if not all(math.isfinite(value) for value in vectors[key]):
                        raise ValueError("Embedding cache contains non-finite values")
            if missing:
                encoded = self._encode(list(missing.values()), "documents")
                if len(encoded) != len(missing):
                    raise ValueError("Embedding output count does not match documents")
                for key, vector in zip(missing, encoded):
                    if len(vector) != self.embedding_dimension:
                        raise ValueError("Embedding output dimension does not match model")
                    if not all(math.isfinite(value) for value in vector):
                        raise ValueError("Embedding output contains non-finite values")
                    blob = struct.pack("<" + "f" * self.embedding_dimension, *vector)
                    db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                               (key, blob, hashlib.sha256(blob).hexdigest()))
                    vectors[key] = vector
                db.commit()
            return [vectors[key] for key in keys]
        except sqlite3.Error as exc:
            raise ConfigError("Embedding cache read/write failed; check free disk space and cache directory permissions. Existing cached vectors are retained.") from exc
        finally:
            if db is not None:
                db.close()

    def _encode(self, texts: list[str], kind: str) -> list[list[float]]:
        # Fast tokenizers mutate truncation/padding state during encode. Protect
        # capacity checks and inference together when a service shares this model.
        with self._encode_lock:
            return self._encode_locked(texts, kind)

    def _encode_locked(self, texts: list[str], kind: str) -> list[list[float]]:
        if kind == "documents":
            prefix = getattr(self, "document_prefix", "")
            texts = [prefix + text for text in texts]
        lengths = [len(self.tokenizer.encode(text)) for text in texts]
        if any(length > self.max_tokens for length in lengths):
            raise ValueError("Embedding text exceeds tokenizer capacity")
        status = "failed"
        try:
            result = self.model.encode(texts, batch_size=getattr(self, "batch_size", 32),
                                       prompt="", normalize_embeddings=True, show_progress_bar=False).tolist()
            status = "completed"
            return result
        finally:
            if self.recorder is not None:
                self.recorder(kind, len(texts), sum(lengths), status)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([getattr(self, "query_prefix", "") + text], "query")[0]


def create_embeddings(config: EmbeddingConfig, cache: Path, *, recorder=None):
    return LocalEmbeddings(cache, config=config, recorder=recorder)
