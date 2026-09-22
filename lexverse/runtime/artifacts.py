"""Manage isolated Trial files and atomic artifact writes."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrialWorkspace:
    trial_dir: Path
    stdout_path: Path
    stderr_path: Path
    status_path: Path
    expected_outputs: tuple[Path, ...] = ()

    @classmethod
    def create(cls, root: Path, trial_id: str, expected_outputs: tuple[str, ...] = ()) -> TrialWorkspace:
        trial_dir = root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        expected = tuple(trial_dir / name for name in expected_outputs)
        return cls(
            trial_dir=trial_dir,
            stdout_path=trial_dir / "stdout.log",
            stderr_path=trial_dir / "stderr.log",
            status_path=trial_dir / "status.json",
            expected_outputs=expected,
        )

    def write_status(self, status: dict) -> None:
        atomic_write_json(self.status_path, status)

    def missing_outputs(self) -> tuple[Path, ...]:
        return tuple(p for p in self.expected_outputs if not p.exists())


@dataclass
class CompletenessReport:
    expected: int
    present: int
    missing: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing and not self.duplicated

    def to_dict(self) -> dict:
        return {
            "expected": self.expected,
            "present": self.present,
            "missing": self.missing,
            "duplicated": self.duplicated,
            "is_complete": self.is_complete,
        }


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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def check_completeness(
    expected_ids: list[str],
    present_ids: list[str],
) -> CompletenessReport:
    expected_set = set(expected_ids)
    present_set = set(present_ids)
    missing = sorted(expected_set - present_set)
    seen: dict[str, int] = {}
    for pid in present_ids:
        seen[pid] = seen.get(pid, 0) + 1
    duplicated = sorted(pid for pid, count in seen.items() if count > 1)
    return CompletenessReport(
        expected=len(expected_set),
        present=len(present_set & expected_set),
        missing=missing,
        duplicated=duplicated,
    )
