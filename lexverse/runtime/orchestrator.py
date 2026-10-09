from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Generic, TypeVar
import asyncio
from collections.abc import Awaitable, Callable
from lexverse.runtime.errors import TrialError
import uuid
import hashlib
import json
import logging
from lexverse.runtime.results import CompletenessReport, atomic_write_json, check_completeness, safe_path_component, trial_result_paths
from lexverse.runtime.results import trial_relative_path
from lexverse.verifiers.base import TaskVerifier, RunEvaluationResult
from lexverse.tasks.schema import LexVerseTask
from lexverse.providers.runtime import scoring_identity


logger = logging.getLogger("lexverse.run")
_PROGRESS_INTERVAL_SEC = 30.0


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


_T = TypeVar("_T")
_R = TypeVar("_R")


TrialFactory = Callable[[str, Path], Awaitable[dict[str, Any]]]
"""Execute one sample; return artifact metadata or raise on failure."""

EvaluatorFn = Callable[[list[Path]], dict[str, Any]]
"""Evaluate successful native outputs when job evaluation is enabled."""


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
    run_hash: str | None = None

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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["work_dir"] = str(self.work_dir)
        if self.native_output_path is not None:
            data["native_output_path"] = str(self.native_output_path)
        if self.evaluation_path is not None:
            data["evaluation_path"] = str(self.evaluation_path)
        return data


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    min_wait_sec: float = 1.0
    max_wait_sec: float = 30.0
    wait_multiplier: float = 2.0
    retry_on: tuple[type[BaseException], ...] = (TrialError,)

    def backoff(self, attempt_index: int) -> float:
        delay = self.min_wait_sec * (self.wait_multiplier ** attempt_index)
        return min(delay, self.max_wait_sec)


@dataclass
class JobSpec:
    benchmark: str
    sample_ids: list[str]
    run_dir: Path
    resolved_config: dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 1
    config_hash: str | None = None
    trial_paths: dict[str, str] = field(default_factory=dict)
    run_hash: str | None = None


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


def _safe_id(s: str) -> str:
    return safe_path_component(s)


def _trial_dir(
    trials_root: Path, sample_id: str, relative_path: str | None = None
) -> Path:
    return trials_root / (relative_path or _safe_id(sample_id))


def _new_job_id() -> str:
    return f"job-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _write_state(
    trial_dir: Path, record: TrialRecord, *, payload: dict[str, Any] | None = None
) -> None:
    data = dict(payload or {})
    data.update({
        "schema_version": 1.0,
        "id": data.get("id", record.task.task_id),
        "benchmark": data.get("benchmark", record.task.benchmark),
        "trial_id": record.trial_id,
        "attempt": record.attempt,
        "status": record.status,
        "config_hash": record.config_hash,
        "run_hash": record.run_hash,
        "work_dir": str(record.work_dir),
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
    })
    atomic_write_json(trial_dir / "results.json", data)


def _scoring_context(value):
    """Hash evaluator inputs without persisting credentials or runtime objects."""
    if isinstance(value, dict):
        ignored = {"concurrency", "on_progress", "resume_native",
                   "resume_task_ids", "resume_artifact_hashes"}
        if "scoring_provider" in value:
            ignored.add("scoring_api_base")
        return {key: _scoring_context(item) for key, item in sorted(value.items())
                if "api_key" not in key.lower() and key not in ignored}
    if isinstance(value, (list, tuple)):
        return [_scoring_context(item) for item in value]
    if hasattr(value, "generate"):
        return scoring_identity(value)
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            if path.is_file():
                return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        except OSError:
            pass
        return str(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return type(value).__name__


def _evaluation_identity(tasks, records, scoring, run_dir):
    inputs = []
    for task in tasks:
        record = records.get(task.id, {})
        artifacts = record.get("execution_artifacts", record.get("artifacts", {}))
        # Older records may mix evaluation and generation artifacts.
        evaluation_paths = set((record.get("evaluation") or {}).get("artifacts", {}).values())
        files = {}
        for key, filename in artifacts.items():
            if filename not in evaluation_paths:
                path = run_dir / filename
                files[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        inputs.append({"task": task.model_dump(mode="json"), "status": record.get("status"),
                       "response": record.get("response"), "artifacts": files})
    value = {"inputs": inputs, "scoring": scoring}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _evaluation_artifacts(result, run_dir):
    artifacts = {}
    for item in [*result.groups, *result.trials.values()]:
        paths = item.get("artifacts", {}) if isinstance(item, dict) else item.artifacts
        for filename in paths.values():
            path = run_dir / filename
            artifacts[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts


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
            run_hash=data.get("run_hash"),
            status=("completed" if data.get("error") == "official evaluator returned no case result"
                    and data.get("response") is not None else data.get("status", "pending")),
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


def _synth_failed(
    sample_id: str,
    benchmark: str,
    trials_root: Path,
    exc: BaseException,
    config_hash: str,
    relative_path: str | None = None,
    run_hash: str | None = None,
) -> TrialRecord:
    trial_dir = _trial_dir(trials_root, sample_id, relative_path)
    trial_dir.mkdir(parents=True, exist_ok=True)
    record = TrialRecord(
        trial_id=f"{_safe_id(sample_id)}-failed",
        task=TaskRecord(benchmark=benchmark, task_id=sample_id, source_ref=""),
        attempt=1,
        work_dir=trial_dir,
        config_hash=config_hash,
        run_hash=run_hash,
    )
    record.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")
    _write_state(trial_dir, record)
    return record


async def _execute_one(
    *,
    sample_id: str,
    benchmark: str,
    trials_root: Path,
    trial_factory: TrialFactory,
    max_attempts: int,
    config_hash: str,
    prior: TrialRecord | None,
    run_hash: str | None = None,
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
            run_hash=run_hash,
        )
        record.mark_started()
        _write_state(trial_dir, record)
        started = time.monotonic()
        logger.info("[task] start task=%s attempt=%s", sample_id, attempt)

        try:
            result = await trial_factory(sample_id, trial_dir)
        except TrialError as exc:
            last_exc = exc
            record.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")
            _write_state(trial_dir, record)
            if attempt >= starting_attempt + max_attempts - 1:
                logger.warning("[task] failed task=%s attempt=%s error=%s elapsed=%.1fs",
                               sample_id, attempt, type(exc).__name__, time.monotonic() - started)
                return record
            logger.warning("[task] retry task=%s attempt=%s/%s error=%s",
                           sample_id, attempt + 1, starting_attempt + max_attempts - 1,
                           type(exc).__name__)
            continue
        except BaseException as exc:
            record.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")
            _write_state(trial_dir, record)
            logger.warning("[task] failed task=%s attempt=%s error=%s elapsed=%.1fs",
                           sample_id, attempt, type(exc).__name__, time.monotonic() - started)
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
        logger.info("[task] completed task=%s attempt=%s elapsed=%.1fs",
                    sample_id, attempt, time.monotonic() - started)
        return record

    assert last_exc is not None
    raise last_exc


class Scheduler(Generic[_T, _R]):
    def __init__(self, n_concurrent: int, retry: RetryPolicy | None = None):
        if n_concurrent < 1:
            raise ValueError("n_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(n_concurrent)
        self._retry = retry or RetryPolicy()

    async def run(
        self,
        items: list[_T],
        worker: Callable[[_T], Awaitable[_R]],
    ) -> list[_R | BaseException]:
        """Return results or exceptions in input order."""
        return await asyncio.gather(
            *(self._one(item, worker) for item in items),
            return_exceptions=True,
        )

    async def _one(
        self,
        item: _T,
        worker: Callable[[_T], Awaitable[_R]],
    ) -> _R:
        async with self._sem:
            last_exc: BaseException | None = None
            for attempt in range(self._retry.max_attempts):
                try:
                    return await worker(item)
                except self._retry.retry_on as exc:
                    last_exc = exc
                    if attempt + 1 >= self._retry.max_attempts:
                        raise
                    await asyncio.sleep(self._retry.backoff(attempt))
            assert last_exc is not None
            raise last_exc


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

        for task_id, record in loaded.items():
            if task_id in job.sample_ids and (
                record.config_hash != config_hash or record.task.benchmark != job.benchmark
                or record.run_hash != job.run_hash
            ):
                raise TrialError("resume configuration/source mismatch; use a new run directory")

        started = time.monotonic()
        counts = {"finished": 0, "success": 0, "failed": 0, "running": 0, "skipped": 0}

        def report_progress() -> None:
            logger.info(
                "[progress] finished=%s/%s success=%s failed=%s running=%s skipped=%s elapsed=%.1fs",
                counts["finished"], len(job.sample_ids), counts["success"],
                counts["failed"], counts["running"], counts["skipped"], time.monotonic() - started,
            )

        async def _run_trial(sample_id: str) -> TrialRecord:
            existing = loaded.get(sample_id)
            if existing is not None and existing.is_successful():
                state_path = _trial_dir(trials_root, sample_id, job.trial_paths.get(sample_id)) / "results.json"
                state = json.loads(state_path.read_text())
                artifacts = state.get("execution_artifacts", {})
                if all((job.run_dir / path).is_file() for path in artifacts.values()):
                    counts["skipped"] += 1
                    logger.info("[task] skipped task=%s reason=already_completed", sample_id)
                    return existing
            return await _execute_one(
                sample_id=sample_id,
                benchmark=job.benchmark,
                trials_root=trials_root,
                trial_factory=trial_factory,
                max_attempts=job.max_attempts,
                config_hash=config_hash,
                run_hash=job.run_hash,
                prior=existing,
                relative_path=job.trial_paths.get(sample_id),
            )

        async def _worker(sample_id: str) -> TrialRecord:
            counts["running"] += 1
            try:
                record = await _run_trial(sample_id)
            except BaseException:
                counts["failed"] += 1
                raise
            else:
                counts["success" if record.is_successful() else "failed"] += 1
                return record
            finally:
                counts["running"] -= 1
                counts["finished"] += 1
                report_progress()

        stopped = asyncio.Event()

        async def heartbeat() -> None:
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=_PROGRESS_INTERVAL_SEC)
                except asyncio.TimeoutError:
                    report_progress()

        progress_task = asyncio.create_task(heartbeat())
        report_progress()
        try:
            outcomes = await self.scheduler.run(job.sample_ids, _worker)
        finally:
            stopped.set()
            await progress_task

        trials: list[TrialRecord] = []
        for sid, res in zip(job.sample_ids, outcomes):
            if isinstance(res, BaseException):
                # Scheduler-level failure (retry exhausted) — synthesize a record.
                trials.append(
                    _synth_failed(
                        sid, job.benchmark, trials_root, res, config_hash,
                        relative_path=job.trial_paths.get(sid),
                        run_hash=job.run_hash,
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

    async def evaluate(
        self, tasks: list[LexVerseTask], records: dict[str, dict], *,
        verifier: TaskVerifier, run_dir: Path, model_name: str, context: dict,
        judge_config: dict, on_progress: Callable[[RunEvaluationResult], None] | None = None,
    ) -> RunEvaluationResult:
        """Recover native scoring units from the single evaluation.json ledger."""
        run_dir = run_dir.resolve()
        path = run_dir / "evaluation.json"
        try:
            previous = json.loads(path.read_text()).get("units", {})
            if not isinstance(previous, dict):
                previous = {}
        except (OSError, ValueError, AttributeError):
            previous = {}
        units = verifier.evaluation_units(tasks)
        checkpoints, results, identities, resume_native = {}, {}, {}, {}
        task_identities, native_contexts = {}, {}
        cached = set()
        written = {}
        running = 0
        scoring = {"verifier": verifier.scoring_identity(), "judge": judge_config,
                   "context": _scoring_context(context), "model": model_name,
                   "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

        def save_records(evaluated):
            for task in tasks:
                record = records.get(task.id, {})
                if record.get("status") != "completed":
                    continue
                result = evaluated.trials.get(task.id)
                old_artifacts = (record.get("evaluation") or {}).get("artifacts", {})
                for name in old_artifacts:
                    if name not in record.get("execution_artifacts", {}):
                        record.get("artifacts", {}).pop(name, None)
                data = (result.model_dump(mode="json") if result is not None else
                        None if verifier.grouped else {"status": "pending", "metrics": {}})
                if data is None:
                    record.pop("evaluation", None)
                else:
                    record["evaluation"] = data
                if result is not None:
                    record.setdefault("artifacts", {}).update(result.artifacts)
                if task.id in written and written[task.id] == data:
                    continue
                # Judge failures never change the generation status or answer.
                directory = run_dir / "trials" / trial_relative_path(
                    task.source.task_type or verifier.name, task.source.sample_id,
                )
                atomic_write_json(directory / "results.json", record)
                written[task.id] = data

        def publish(notify=True):
            evaluated = verifier.build_result(tasks, results)
            state = evaluated.to_dict()
            state.update(units=checkpoints, cached=len(cached), running=running)
            atomic_write_json(path, state)
            save_records(evaluated)
            if notify and on_progress is not None:
                on_progress(evaluated)
            return evaluated

        for key, unit in units.items():
            identity = _evaluation_identity(unit, records, scoring, run_dir)
            identities[key] = identity
            saved = previous.get(key, {})
            if not isinstance(saved, dict):
                saved = {}
            resume_native[key] = (saved.get("identity") == identity
                                  and saved.get("native_identity") == identity)
            task_identities[key] = {
                task.id: _evaluation_identity([task], records, scoring, run_dir)
                for task in unit
            } if verifier.grouped else {}
            saved_tasks = saved.get("task_identities", {})
            saved_artifacts = saved.get("artifacts", {})
            if not isinstance(saved_tasks, dict):
                saved_tasks = {}
            if not isinstance(saved_artifacts, dict):
                saved_artifacts = {}
            native_contexts[key] = {
                "resume_task_ids": [task_id for task_id, task_identity in task_identities[key].items()
                                    if saved_tasks.get(task_id) == task_identity or resume_native[key]],
                "resume_artifact_hashes": {str((run_dir / filename).resolve()): digest
                                           for filename, digest in saved_artifacts.items()},
            }
            try:
                result = RunEvaluationResult.from_dict(saved["result"])
                valid = (resume_native[key] and saved["status"] == "completed"
                         and result.expected_task_ids == [task.id for task in unit]
                         and result.to_dict()["status"] == "completed"
                         and saved["artifacts"] == _evaluation_artifacts(result, run_dir))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                valid = False
            if valid:
                results[key] = result
                checkpoints[key] = saved
                cached.add(key)
            else:
                if saved.get("status") == "completed":
                    resume_native[key] = False
                checkpoints[key] = {"identity": identity, "status": "pending",
                                    "task_ids": [task.id for task in unit],
                                    "native_identity": saved.get("native_identity") if resume_native[key] else None,
                                    "task_identities": saved_tasks, "artifacts": saved_artifacts}
        publish()

        async def worker(key):
            nonlocal running
            running += 1
            try:
                unit_context = {**context, "resume_native": resume_native[key], **native_contexts[key]}
                verifier.prepare_unit(units[key], records, run_dir=run_dir,
                                      model_name=model_name, context=unit_context)
                checkpoints[key]["artifacts"] = {
                    filename: digest for filename, digest in checkpoints[key].get("artifacts", {}).items()
                    if (run_dir / filename).is_file()
                }
                checkpoints[key].update(status="running", native_identity=identities[key],
                                        task_identities=task_identities[key])
                publish(notify=False)
                result = await verifier.evaluate_unit(
                    units[key], records, run_dir=run_dir, model_name=model_name,
                    context=unit_context,
                )
                result = RunEvaluationResult.from_dict(result.to_dict())
                if result.expected_task_ids != [task.id for task in units[key]]:
                    raise ValueError("native scoring unit returned mismatched task IDs")
                for item in [*result.groups, *result.trials.values()]:
                    artifacts = item.get("artifacts", {}) if isinstance(item, dict) else item.artifacts
                    for name, filename in list(artifacts.items()):
                        artifacts[name] = str(Path(filename).relative_to(run_dir)) if Path(filename).is_absolute() else filename
                status = result.to_dict()["status"]
                checkpoints[key].update(status=status, result=result.to_dict(),
                                        artifacts=_evaluation_artifacts(result, run_dir))
                results[key] = result
                return result
            except Exception as exc:
                checkpoints[key].update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                running -= 1
                publish()

        stopped = asyncio.Event()

        async def heartbeat():
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(stopped.wait(), _PROGRESS_INTERVAL_SEC)
                except asyncio.TimeoutError:
                    publish()

        progress = asyncio.create_task(heartbeat())
        try:
            pending = [key for key, unit in units.items() if key not in cached
                       and any(records.get(task.id, {}).get("status") == "completed" for task in unit)]
            outcomes = await self.scheduler.run(pending, worker)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        finally:
            stopped.set()
            await progress
            evaluated = publish()
        return evaluated
