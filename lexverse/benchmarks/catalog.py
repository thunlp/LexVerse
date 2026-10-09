from __future__ import annotations

import json
import os
from pathlib import Path

from filelock import FileLock
import yaml

from lexverse.runtime.errors import TrialError


def load_catalog(path: Path) -> dict:
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = catalog.get("source") if isinstance(catalog, dict) else None
    if (not isinstance(source, dict) or set(source) != {"name", "repo_url", "commit"}
            or any(not isinstance(value, str) or not value.strip() for value in source.values())):
        raise ValueError(f"catalog source must define name, repo_url and commit: {path}")
    if (catalog.get("schema_version") != 1.0
            or not isinstance(catalog.get("dataset"), dict)
            or not isinstance(catalog.get("tasks"), dict)):
        raise ValueError(f"invalid benchmark catalog: {path}")
    dataset = catalog["dataset"]
    if "paths" in dataset:
        valid = (isinstance(dataset["paths"], list) and bool(dataset["paths"])
                 and all(isinstance(pattern, str) and pattern for pattern in dataset["paths"]))
    else:
        valid = (isinstance(dataset.get("repo_id"), str) and bool(dataset["repo_id"])
                 and isinstance(dataset.get("revision"), str) and len(dataset["revision"]) == 40
                 and all(char in "0123456789abcdef" for char in dataset["revision"]))
    if not valid:
        raise ValueError(f"invalid catalog dataset: {path}")
    for name, task in catalog["tasks"].items():
        if (not isinstance(task, dict) or task.get("output_type") not in {"text", "choice", "trajectory"}
                or not isinstance(task.get("metrics"), list) or not task["metrics"]
                or any(not isinstance(metric, str) or not metric for metric in task["metrics"])):
            raise ValueError(f"invalid catalog task: {name}")
    return catalog


def dataset_files(catalog: dict, *, root: Path | None = None, offline: bool = False) -> dict[str, Path]:
    dataset = catalog["dataset"]
    if "paths" in dataset:
        if root is None:
            raise ValueError("repository dataset requires its upstream root")
        files = {}
        for pattern in dataset["paths"]:
            matches = sorted(path for path in root.glob(pattern) if path.is_file())
            if not matches:
                raise ValueError(f"catalog dataset path has no files: {pattern}")
            files.update({str(path.relative_to(root)): path for path in matches})
        return files

    name = catalog["source"]["name"]
    repo, revision = dataset["repo_id"], dataset["revision"]
    cache = Path(os.environ.get("LEXVERSE_DATASET_CACHE", ".lexverse/datasets")) / name / revision
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / ".lexverse-dataset.json"
    with FileLock(str(cache) + ".lock"):
        names = json.loads(marker.read_text()) if marker.is_file() else []
        if not names or any(not (cache / filename).is_file() for filename in names):
            if offline:
                raise TrialError(f"complete {name} dataset at revision {revision} is not cached: {cache}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                                  allow_patterns=["*.json", "*.jsonl"], local_dir=cache)
            except ImportError as exc:
                raise TrialError(f"{name} dataset download requires the {name} extra") from exc
            except Exception as exc:
                raise TrialError(f"unable to download {repo} at {revision}; for gated data, accept its terms and run `hf auth login`") from exc
            names = sorted(str(path.relative_to(cache)) for path in cache.rglob("*")
                           if path.is_file() and path.suffix in {".json", ".jsonl"}
                           and not any(part.startswith(".") for part in path.relative_to(cache).parts))
            if not names:
                raise TrialError(f"no JSON/JSONL data files in {repo} at {revision}")
            marker.write_text(json.dumps(names), encoding="utf-8")
    return {filename: (cache / filename).resolve() for filename in names}
