"""Atomic result writes and safe names for current native artifacts."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path_component(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in str(value)
    )


def artifact_model_name(name: str) -> str:
    """Keep HF IDs and absolute weight paths inside the run's artifact tree."""
    import re
    if re.fullmatch(r"[\w.-]+", name) and name not in {".", ".."}:
        return name
    readable = re.sub(r"[^\w.-]+", "-", Path(name).name).strip(".-") or "model"
    return readable + "-" + hashlib.sha256(name.encode()).hexdigest()[:8]


def trial_relative_path(task_type: str, sample_id: str) -> Path:
    return Path(safe_path_component(task_type)) / safe_path_component(sample_id)


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via temp file + rename so partial writes never appear."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_jsonl(path: Path):
    """Read native JSON events, ignoring CLI diagnostic lines."""
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event
