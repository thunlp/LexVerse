"""Native scoring checkpoints and recovery for current Job evaluation exports."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
from abc import ABC, abstractmethod
import json
from collections.abc import Callable
from pathlib import Path

from lexverse.llms.runtime import scoring_identity
from lexverse.utils.files import atomic_write_json
from lexverse.utils.identity import identity
from lexverse.job import Scheduler
from lexverse.models.task.task import LexVerseTask
from lexverse.models.verifier.result import RunVerificationResult


class BenchmarkVerifier(ABC):
    """Native scoring adapters sharing one run-level recovery loop."""

    name: str
    grouped = False

    def build_verification_units(self, tasks: list[LexVerseTask]) -> dict[str, list[LexVerseTask]]:
        units = {}
        for task in tasks:
            key = task.source.task_type if self.grouped else task.id
            units.setdefault(key, []).append(task)
        return units

    def scoring_identity(self) -> dict:
        module = importlib.import_module(type(self).__module__)
        directory = Path(module.__file__).parent
        upstream = getattr(importlib.import_module(module.__package__), "UPSTREAM", None)
        root = getattr(self, "root", upstream.pristine_dir if upstream else None)
        files = sorted(directory.rglob("*.py")) + sorted(directory.rglob("*.patch"))
        if root is not None:
            files += sorted(Path(root).rglob("*.py"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(directory) if path.is_relative_to(directory)
                            else path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return {"name": self.name, "upstream_commit": upstream.commit if upstream else None,
                "implementation_sha256": digest.hexdigest()}

    async def verify_run(
        self, tasks: list[LexVerseTask], records: dict[str, dict], *,
        run_dir: Path, model_name: str, context: dict | None = None, legacy: bool = False,
    ) -> RunVerificationResult:
        context = dict(context or {})
        concurrency = context.pop("concurrency", 1)
        on_progress = context.pop("on_progress", None)
        result = await verify_run(
            tasks, records, verifier=self, run_dir=run_dir, model_name=model_name,
            context=context, judge_config=context.get("judge_config", {}), concurrency=concurrency, on_progress=on_progress, legacy=legacy,
        )
        self.write_results(result, tasks, records, run_dir=run_dir, model_name=model_name)
        return result

    @abstractmethod
    async def verify_unit(
        self, tasks: list[LexVerseTask], records: dict[str, dict], *,
        run_dir: Path, model_name: str, context: dict,
    ) -> RunVerificationResult: ...

    def prepare_unit(self, tasks, records, *, run_dir, model_name, context):
        """Adapters with native intermediate caches validate them before scoring."""

    def collect_partial_unit(self, tasks, records, *, run_dir, model_name, context):
        return None

    def build_result(self, tasks, results):
        trials = {key: value for result in results.values() for key, value in result.trials.items()}
        groups = ([group for result in results.values() for group in result.groups]
                  if self.grouped else self.build_groups(tasks, trials))
        return RunVerificationResult(groups=groups, trials=trials,
                                   expected_task_ids=[task.id for task in tasks])

    def build_groups(self, tasks, results):
        groups = []
        for kind in dict.fromkeys(task.source.task_type for task in tasks):
            ids = [task.id for task in tasks if task.source.task_type == kind
                   and task.id in results and results[task.id].status == "completed"]
            if ids:
                common = set.intersection(*(set(results[sid].metrics) for sid in ids))
                groups.append({"task_type": kind, "task_ids": ids, "metrics": {
                    key: sum(results[sid].metrics[key] for sid in ids) / len(ids)
                    for key in sorted(common)}})
        return groups

    def write_results(self, result, tasks, records, *, run_dir, model_name):
        """Adapters with native batch reports write them after recovery and merging."""


_PROGRESS_INTERVAL_SEC = 30.0


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


def _compute_verification_identity(tasks, records, scoring, run_dir):
    inputs = []
    for task in tasks:
        record = records.get(task.id, {})
        files = {}
        for key, filename in record.get("execution_artifacts", {}).items():
            path = run_dir / filename
            files[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        inputs.append({"task": task.model_dump(mode="json"), "status": record.get("status"),
                       "response": record.get("response"), "artifacts": files,
                       "execution_artifact_hashes": record.get("execution_artifact_hashes")})
    value = {"inputs": inputs, "scoring": scoring}
    return identity(value)


def _hash_verification_artifacts(result, run_dir):
    artifacts = {}
    for item in [*result.groups, *result.trials.values()]:
        paths = item.get("artifacts", {}) if isinstance(item, dict) else item.artifacts
        for filename in paths.values():
            path = run_dir / filename
            artifacts[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts


async def verify_run(
    tasks: list[LexVerseTask], records: dict[str, dict], *,
    verifier: BenchmarkVerifier, run_dir: Path, model_name: str, context: dict,
    judge_config: dict, concurrency: int = 1, on_progress: Callable[[RunVerificationResult], None] | None = None, legacy: bool = False,
) -> RunVerificationResult:
    """Recover native scoring units from the single evaluation.json ledger."""
    run_dir = run_dir.resolve()
    path = run_dir / ("evaluation.json" if legacy else "verification.json")
    try:
        previous = json.loads(path.read_text()).get("units", {})
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError, AttributeError):
        previous = {}
    units = verifier.build_verification_units(tasks)
    checkpoints, results, identities, resume_native = {}, {}, {}, {}
    task_identities, native_contexts = {}, {}
    cached = set()
    running = 0
    scoring = {"verifier": verifier.scoring_identity(), "judge": judge_config,
               "context": _scoring_context(context), "model": model_name,
               "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def update_records(verified):
        for task in tasks:
            record = records.get(task.id, {})
            if record.get("status") != "completed":
                continue
            result = verified.trials.get(task.id)
            old_artifacts = (record.get("verifier_result") or {}).get("artifacts", {})
            for name in old_artifacts:
                if name not in record.get("execution_artifacts", {}):
                    record.get("artifacts", {}).pop(name, None)
            data = (result.model_dump(mode="json") if result is not None else
                    None if verifier.grouped else {"status": "pending", "metrics": {}})
            if data is None:
                record.pop("verifier_result", None)
            else:
                record["verifier_result"] = data
            if result is not None:
                record.setdefault("artifacts", {}).update(result.artifacts)

    def publish(notify=True):
        verified = verifier.build_result(tasks, results)
        state = verified.to_dict()
        if any(unit.get("status") == "incomplete" for unit in checkpoints.values()):
            state["status"] = "incomplete"
        state.update(units=checkpoints, cached=len(cached), running=running)
        atomic_write_json(path, state)
        update_records(verified)
        if notify and on_progress is not None:
            on_progress(verified)
        return verified

    for key, unit in units.items():
        identity = _compute_verification_identity(unit, records, scoring, run_dir)
        identities[key] = identity
        saved = previous.get(key, {})
        if not isinstance(saved, dict):
            saved = {}
        resume_native[key] = (saved.get("identity") == identity
                              and saved.get("native_identity") == identity)
        task_identities[key] = {
            task.id: _compute_verification_identity([task], records, scoring, run_dir)
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
            result = RunVerificationResult.from_dict(saved["result"])
            valid = (resume_native[key] and saved["status"] == "completed"
                     and result.expected_task_ids == [task.id for task in unit]
                     and result.to_dict()["status"] == "completed"
                     and saved["artifacts"] == _hash_verification_artifacts(result, run_dir))
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

    def save_unit(key, result, *, partial=False):
        result = RunVerificationResult.from_dict(result.to_dict())
        if result.expected_task_ids != [task.id for task in units[key]]:
            raise ValueError("native scoring unit returned mismatched task IDs")
        for item in [*result.groups, *result.trials.values()]:
            artifacts = item.get("artifacts", {}) if isinstance(item, dict) else item.artifacts
            for name, filename in list(artifacts.items()):
                artifacts[name] = str(Path(filename).relative_to(run_dir)) if Path(filename).is_absolute() else filename
        status = "incomplete" if partial else result.to_dict()["status"]
        checkpoints[key].update(status=status, result=result.to_dict(),
                        artifacts=_hash_verification_artifacts(result, run_dir))
        results[key] = result
        return result

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
            result = await verifier.verify_unit(
                units[key], records, run_dir=run_dir, model_name=model_name,
                context=unit_context,
            )
            return save_unit(key, result)
        except asyncio.CancelledError:
            checkpoints[key].update(status="incomplete")
            try:
                partial = verifier.collect_partial_unit(units[key], records, run_dir=run_dir,
                                                        model_name=model_name, context=unit_context)
                if partial is not None:
                    save_unit(key, partial, partial=True)
            except Exception as exc:
                checkpoints[key].update(partial_error_type=type(exc).__name__)
            raise
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
        outcomes = await Scheduler(n_concurrent=concurrency).run(pending, worker)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
    finally:
        stopped.set()
        await progress
        verified = publish()
    return verified
