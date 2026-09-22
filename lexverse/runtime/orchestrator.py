"""Run checkpointed Trials with bounded concurrency and retry."""
from __future__ import annotations

import time
import uuid
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .artifacts import CompletenessReport, atomic_write_json, check_completeness
from .errors import TrialError
from .records import TaskRecord, TrialRecord
from .scheduler import RetryPolicy, Scheduler
from .paths import safe_path_component, trial_result_paths


TrialFactory = Callable[[str, Path], Awaitable[dict[str, Any]]]
"""Callable(sample_id, trial_work_dir) -> trial execution result dict.

The factory is what the Adapter provides. On success it returns a dict
of per-trial artifact info (paths, counts, ...). On failure it raises;
the orchestrator captures the exception into the TrialRecord.
"""

EvaluatorFn = Callable[[list[Path]], dict[str, Any]]
"""Callable(list_of_native_output_paths) -> evaluator result dict.

Called once at the end of a Job over all successfully-executed trials.
Only invoked if `evaluate=True` was passed.
"""


@dataclass
class JobSpec:
    benchmark: str
    sample_ids: list[str]
    run_dir: Path
    resolved_config: dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 1
    config_hash: str | None = None
    trial_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class JobResult:
    job_id: str
    run_dir: Path
    benchmark: str
    trials: list[TrialRecord]
    completeness: CompletenessReport
    evaluation: dict[str, Any] | None = None

    def successful_trials(self) -> list[TrialRecord]:
        return [t for t in self.trials if t.is_successful()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_dir": str(self.run_dir),
            "benchmark": self.benchmark,
            "trials": [t.to_dict() for t in self.trials],
            "completeness": self.completeness.to_dict(),
            "evaluation": self.evaluation,
        }


class Orchestrator:
    def __init__(
        self,
        n_concurrent: int = 1,
        retry: RetryPolicy | None = None,
    ):
        self.scheduler = Scheduler[str, TrialRecord](
            n_concurrent=n_concurrent, retry=retry
        )

    async def run(
        self,
        job: JobSpec,
        trial_factory: TrialFactory,
        *,
        resume: bool = False,
    ) -> JobResult:
        job_id = _new_job_id()
        job.run_dir.mkdir(parents=True, exist_ok=True)
        trials_root = job.run_dir / "trials"
        trials_root.mkdir(exist_ok=True)

        loaded: dict[str, TrialRecord] = {}
        if resume:
            loaded = _load_existing_trials(trials_root, job.benchmark)

        config_hash = job.config_hash or hashlib.sha256(
            json.dumps(job.resolved_config, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

        async def _worker(sample_id: str) -> TrialRecord:
            existing = loaded.get(sample_id)
            if existing is not None and existing.is_successful():
                return existing
            return await _execute_one(
                sample_id=sample_id,
                benchmark=job.benchmark,
                trials_root=trials_root,
                trial_factory=trial_factory,
                max_attempts=job.max_attempts,
                config_hash=config_hash,
                prior=existing,
                relative_path=job.trial_paths.get(sample_id),
            )

        outcomes = await self.scheduler.run(job.sample_ids, _worker)

        trials: list[TrialRecord] = []
        for sid, res in zip(job.sample_ids, outcomes):
            if isinstance(res, BaseException):
                # Scheduler-level failure (retry exhausted) — synthesize a record.
                trials.append(
                    _synth_failed(
                        sid, job.benchmark, trials_root, res, config_hash,
                        relative_path=job.trial_paths.get(sid),
                    )
                )
            else:
                trials.append(res)

        completeness = check_completeness(
            expected_ids=job.sample_ids,
            present_ids=[t.task.task_id for t in trials if t.is_successful()],
        )

        return JobResult(
            job_id=job_id,
            run_dir=job.run_dir,
            benchmark=job.benchmark,
            trials=trials,
            completeness=completeness,
        )


# -------------------------------------------------------------------------

def _new_job_id() -> str:
    return f"job-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _trial_dir(
    trials_root: Path, sample_id: str, relative_path: str | None = None
) -> Path:
    return trials_root / (relative_path or _safe_id(sample_id))


def _safe_id(s: str) -> str:
    # Filesystem-safe: keep letters/digits/dashes/underscores, replace the rest.
    return safe_path_component(s)


def _load_existing_trials(trials_root: Path, benchmark: str) -> dict[str, TrialRecord]:
    out: dict[str, TrialRecord] = {}
    for state_path in trial_result_paths(trials_root):
        import json
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_value = data.get("task")
        task_data = task_value if isinstance(task_value, dict) else {}
        task_id = str(data.get("id") or task_data.get("task_id") or "")
        record = TrialRecord(
            trial_id=data["trial_id"],
            task=TaskRecord(
                benchmark=data.get("benchmark", task_data.get("benchmark", benchmark)),
                task_id=task_id,
                source_ref=task_data.get("source_ref", ""),
                metadata=task_data.get("metadata", {}),
            ),
            attempt=data.get("attempt", 1),
            work_dir=Path(data["work_dir"]),
            config_hash=data.get("config_hash", ""),
            status=data.get("status", "pending"),
            eval_status=(
                "scored"
                if (data.get("evaluation") or {}).get("status") == "completed"
                else data.get("eval_status", "pending")
            ),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            native_output_path=(
                Path(data["native_output_path"])
                if data.get("native_output_path")
                else None
            ),
            evaluation_path=(
                Path(data["evaluation_path"])
                if data.get("evaluation_path")
                else None
            ),
        )
        out[task_id or record.trial_id] = record
    return out


async def _execute_one(
    *,
    sample_id: str,
    benchmark: str,
    trials_root: Path,
    trial_factory: TrialFactory,
    max_attempts: int,
    config_hash: str,
    prior: TrialRecord | None,
    relative_path: str | None = None,
) -> TrialRecord:
    trial_dir = _trial_dir(trials_root, sample_id, relative_path)
    trial_dir.mkdir(parents=True, exist_ok=True)

    starting_attempt = 1
    if prior is not None:
        starting_attempt = prior.attempt + 1

    last_exc: BaseException | None = None
    for attempt in range(starting_attempt, starting_attempt + max_attempts):
        record = TrialRecord(
            trial_id=f"{_safe_id(sample_id)}-a{attempt}",
            task=TaskRecord(
                benchmark=benchmark, task_id=sample_id, source_ref=""
            ),
            attempt=attempt,
            work_dir=trial_dir,
            config_hash=config_hash,
        )
        record.mark_started()
        _write_state(trial_dir, record)

        try:
            result = await trial_factory(sample_id, trial_dir)
        except TrialError as exc:
            last_exc = exc
            record.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")
            _write_state(trial_dir, record)
            if attempt >= starting_attempt + max_attempts - 1:
                return record
            continue
        except BaseException as exc:
            record.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")
            _write_state(trial_dir, record)
            raise

        record.mark_finished("completed")
        if isinstance(result, dict) and "output_path" in result:
            record.native_output_path = Path(result["output_path"])
        if isinstance(result, dict) and "save_path" in result:
            record.native_output_path = Path(result["save_path"])
        payload = result.get("trial_result") if isinstance(result, dict) else None
        if isinstance(payload, dict):
            evaluation = payload.get("evaluation") or {}
            record.eval_status = (
                "scored" if evaluation.get("status") == "completed" else "failed"
            )
            verifier_path = trial_dir / "verifier" / "result.json"
            if verifier_path.exists():
                record.evaluation_path = verifier_path
        _write_state(trial_dir, record, payload=payload)
        return record

    assert last_exc is not None
    raise last_exc


def _synth_failed(
    sample_id: str,
    benchmark: str,
    trials_root: Path,
    exc: BaseException,
    config_hash: str,
    relative_path: str | None = None,
) -> TrialRecord:
    trial_dir = _trial_dir(trials_root, sample_id, relative_path)
    trial_dir.mkdir(parents=True, exist_ok=True)
    record = TrialRecord(
        trial_id=f"{_safe_id(sample_id)}-failed",
        task=TaskRecord(benchmark=benchmark, task_id=sample_id, source_ref=""),
        attempt=1,
        work_dir=trial_dir,
        config_hash=config_hash,
    )
    record.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")
    _write_state(trial_dir, record)
    return record


def _write_state(
    trial_dir: Path, record: TrialRecord, *, payload: dict[str, Any] | None = None
) -> None:
    data = dict(payload or {})
    data.update({
        "schema_version": 1,
        "id": data.get("id", record.task.task_id),
        "benchmark": data.get("benchmark", record.task.benchmark),
        "trial_id": record.trial_id,
        "attempt": record.attempt,
        "status": record.status,
        "config_hash": record.config_hash,
        "work_dir": str(record.work_dir),
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
    })
    atomic_write_json(trial_dir / "results.json", data)
