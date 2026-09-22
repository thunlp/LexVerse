"""Runtime records kept separate from upstream benchmark payloads."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal


TrialStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "timeout",
    "cancelled",
]

EvalStatus = Literal[
    "pending",
    "running",
    "scored",
    "failed",
    "skipped",
]


@dataclass(frozen=True)
class TaskRecord:
    benchmark: str
    task_id: str
    source_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialRecord:
    trial_id: str
    task: TaskRecord
    attempt: int
    work_dir: Path
    config_hash: str
    status: TrialStatus = "pending"
    eval_status: EvalStatus = "pending"
    exit_code: int | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    native_output_path: Path | None = None
    evaluation_path: Path | None = None

    def mark_started(self) -> None:
        self.status = "running"
        self.started_at = time.time()

    def mark_finished(self, status: TrialStatus, *, exit_code: int | None = None, error: str | None = None) -> None:
        self.status = status
        self.exit_code = exit_code
        self.error = error
        self.finished_at = time.time()

    def is_successful(self) -> bool:
        """Execution completed cleanly. Evaluation status is tracked separately."""
        return self.status == "completed"

    def failed_from_exception(self, exc: BaseException) -> None:
        self.status = "failed"
        self.error = f"{type(exc).__name__}: {exc}"
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["work_dir"] = str(self.work_dir)
        if self.native_output_path is not None:
            data["native_output_path"] = str(self.native_output_path)
        if self.evaluation_path is not None:
            data["evaluation_path"] = str(self.evaluation_path)
        return data


@dataclass
class JobRecord:
    job_id: str
    benchmark: str
    run_dir: Path
    resolved_config_path: Path
    selected_samples: list[TaskRecord]
    concurrency: int
    max_attempts: int
    expected_trial_ids: list[str]
    completeness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "benchmark": self.benchmark,
            "run_dir": str(self.run_dir),
            "resolved_config_path": str(self.resolved_config_path),
            "selected_samples": [s.to_dict() for s in self.selected_samples],
            "concurrency": self.concurrency,
            "max_attempts": self.max_attempts,
            "expected_trial_ids": list(self.expected_trial_ids),
            "completeness": dict(self.completeness),
        }
