from __future__ import annotations

from pathlib import Path


def safe_path_component(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in str(value)
    )


def trial_relative_path(task_type: str, sample_id: str) -> Path:
    return Path(safe_path_component(task_type)) / safe_path_component(sample_id)


def trial_result_paths(trials_root: Path) -> list[Path]:
    """List canonical Trial results plus legacy singular filenames."""
    paths = {
        *trials_root.glob("*/results.json"),
        *trials_root.glob("*/*/results.json"),
        *trials_root.glob("*/result.json"),
        *trials_root.glob("*/*/result.json"),
    }
    return sorted(path for path in paths if path.parent.name != "verifier")
