"""Resolve the gated J1-Eval dataset into LexVerse's local cache."""
from __future__ import annotations

from pathlib import Path

from lexverse.runtime.errors import TrialError


DATASET_REPO_ID = "CimoInkPool/J1-Eval_Dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
DEFAULT_DATASET_DIR = Path(".lexverse/datasets/j1bench")


def resolve_case_database(scenario: str, *, offline: bool = False) -> Path:
    """Return a cached scenario file, downloading it on first use.

    Authentication is intentionally delegated to ``huggingface_hub`` so the
    normal ``hf auth login`` token store is used and no token enters LexVerse
    configuration or run artifacts.
    """
    filename = f"J1-Eval_{scenario}.jsonl"
    destination = DEFAULT_DATASET_DIR / filename
    if destination.is_file():
        return destination

    if offline:
        raise TrialError(
            f"J1Bench dataset is not cached at {destination}. Disable "
            "execution.offline for the first prepare, or download it manually."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise TrialError(
            "J1Bench automatic dataset download requires the j1bench extra: "
            "python -m pip install -e '.[j1bench]'"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=destination.parent,
        )
    except Exception as exc:
        raise TrialError(
            "Unable to download the gated J1Bench dataset. First accept the "
            f"dataset terms at {DATASET_URL}, then run `hf auth login` and retry. "
            f"Requested file: {filename}. Original error: {exc}"
        ) from exc

    return Path(downloaded)
