"""Validate and fingerprint optional native law retrieval resources."""
from __future__ import annotations

import json
from pathlib import Path

from lexverse.config import ConfigError


def law_resources(config):
    settings = config.benchmark.get("retrieval", {}).get("laws", {})
    if not settings.get("enabled", False):
        return {}, {}
    import numpy as np

    root = Path(settings.get("index_dir", "")).expanduser()
    if not root.is_absolute():
        raise ConfigError("legalworld retrieval.laws.index_dir must be an absolute path")
    manifest_path = root / "law_vector_index_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        vectors = root / manifest.get("vector_file", "law_embeddings.float16.npy")
        metadata = root / manifest.get("metadata_file", "law_metadata.jsonl")
        matrix = np.load(vectors, mmap_mode="r", allow_pickle=False)
        records = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("invalid law metadata")
        count = len(records)
        dimensions = int(manifest.get("vector_dim") or matrix.shape[1])
        if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape != (count, dimensions):
            raise ValueError("law index shape/count mismatch")
        model = settings["embedding_model"]
        profile = settings["embedding_profile"]
        hint = manifest.get("query_embedding_model_hint")
        if not hint or hint != model:
            raise ValueError("embedding model differs from index model hint")
        if not model or not profile:
            raise ValueError("embedding model and profile are required")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise ConfigError(f"invalid Legal-world law index: {type(exc).__name__}") from exc
    return {"index_dir": str(root.resolve()), "embedding_model": model,
            "embedding_profile": profile, "dimensions": dimensions}, {
        "law_index_manifest": manifest_path, "law_vectors": vectors, "law_metadata": metadata,
    }


def case_resources(config):
    settings = config.benchmark.get("retrieval", {}).get("cases", {})
    if not settings.get("enabled", False):
        return {}, {}
    path = Path(settings.get("docs_path", "")).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ConfigError("legalworld retrieval.cases.docs_path must be an existing absolute path")
    if not settings.get("source_url") or not settings.get("revision"):
        raise ConfigError("case retrieval requires source_url and revision provenance")
    try:
        docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not docs or any(not isinstance(doc, dict) or not all(key in doc for key in ("case_id", "title_clean", "case_cause", "legal_basis")) or not any(doc.get(key) for key in
                           ("title_clean", "first_instance_text", "second_instance_text")) for doc in docs):
            raise ValueError("invalid native case corpus")
    except (OSError, ValueError) as exc:
        raise ConfigError("invalid Legal-world case retrieval JSONL") from exc
    return {"docs_path": str(path.resolve()), "source_url": settings["source_url"],
            "revision": settings["revision"]}, {"case_retrieval_docs": path}


def exclude_benchmark_cases(docs, benchmark):
    """Conservatively exclude matching docket numbers and near-duplicate long texts."""
    import re

    def leaves(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if "case_number" in key and isinstance(item, str) and item.strip():
                    numbers.add(item.strip())
                yield from leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from leaves(item)
        elif isinstance(value, str):
            text = re.sub(r"\s+", "", value)
            if len(text) >= 200:
                yield {text[i:i + 20] for i in range(len(text) - 19)}

    numbers = set()
    references = list(leaves(benchmark))
    kept, excluded = [], []
    for index, doc in enumerate(docs):
        if any(number in json.dumps(doc, ensure_ascii=False) for number in numbers):
            excluded.append(index)
            continue
        texts = list(leaves({key: doc.get(key, "") for key in ("first_instance_text", "second_instance_text")}))
        if any(len(text & reference) / min(len(text), len(reference)) >= .8
               for text in texts for reference in references):
            excluded.append(index)
        else:
            kept.append(doc)
    return kept, excluded
