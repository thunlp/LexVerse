"""Job planning, bounded scheduling, and aggregate reports."""
from __future__ import annotations

import asyncio
from collections import Counter
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Generic, TypeVar
from urllib.parse import quote

from lexverse.config import ConfigError
from lexverse.models.job.result import JobResult
from lexverse.models.trial.config import ExecutionConfig
from lexverse.trial.trial import Trial, TrialExecutor
from lexverse.utils.identity import identity
from lexverse.utils.files import atomic_write_json
from lexverse.models.task.task import LexVerseTask
from lexverse.models.verification_compat import convert_task_names, convert_result_names


def require_compatible(task: LexVerseTask, execution: ExecutionConfig) -> None:
    from lexverse.agents.factory import validate_agent
    if task.source.kind == 'user':
        from lexverse.tasks.user import SUPPORTED_AGENTS
        allowed = SUPPORTED_AGENTS
    else:
        from lexverse.benchmarks.registry import get_plugin
        allowed = get_plugin(task.source.name).supported_agents
    config = execution.settings.get('config', {})
    capabilities = config.get('resources', {})
    environment = execution.settings.get('environment', config.get('environment', {}).get('type', 'docker'))
    validate_agent(execution.agent, allowed, source=task.source.name, model=execution.model,
                     environment=environment,
                     network_mode=task.environment.get('network_mode', capabilities.get('network_mode', 'public')))


@dataclass
class Job:
    tasks: list[LexVerseTask]
    executions: list[ExecutionConfig]
    directory: Path
    n_attempts: int = 1
    concurrency: int = 1
    max_retries: int = 0
    id: str = ""
    schema_version: str = "1.1"

    def __post_init__(self):
        self.directory = self.directory.expanduser().resolve()
        if not self.id:
            self.id = self.directory.name

    @property
    def legacy(self):
        return self.schema_version == "1.0"

    @property
    def verification_directory(self):
        return self.directory / ("evaluation" if self.legacy else "verification")

    @property
    def verification_path(self):
        return self.verification_directory / ("evaluation.json" if self.legacy else "verification.json")

    def build_verification_ref(self, fragment):
        return f"{self.verification_path.relative_to(self.directory)}#{fragment}"

    def plan(self) -> list[Trial]:
        if not self.tasks or not self.executions:
            raise ConfigError('A Job needs tasks and execution configurations')
        if any(type(value) is not int or value < minimum for value, minimum in
               [(self.n_attempts, 1), (self.concurrency, 1), (self.max_retries, 0)]):
            raise ConfigError('attempts/concurrency must be positive integers; retries must be nonnegative')
        if len({task.id for task in self.tasks}) != len(self.tasks):
            raise ConfigError('Duplicate task IDs')
        if len({execution.id for execution in self.executions}) != len(self.executions):
            raise ConfigError('Duplicate execution configurations')
        for task in self.tasks:
            for execution in self.executions:
                require_compatible(task, execution)
        trials = []
        for attempt in range(1, self.n_attempts + 1):
            for task in self.tasks:
                case = (f'{task.source.task_type or task.source.name}_case-{task.source.sample_id}'
                        if task.source.kind == 'benchmark' else task.id)
                for execution in self.executions:
                    name = (f"{quote(case, safe='')}__{quote(execution.model['name'], safe='')}"
                            f"__a{attempt}__{execution.id.removeprefix('config-')[:8]}")
                    trials.append(Trial(
                        id=f'trial-{identity([self.id, task.id, execution.id, attempt])[:20]}', task=task,
                        execution=execution, attempt=attempt,
                        directory=self.directory / 'trials' / quote(execution.agent, safe='') / name,
                        job_directory=self.directory,
                    ))
        if len({trial.directory for trial in trials}) != len(trials):
            raise ConfigError('Trial directory names collide; use distinct case IDs or execution configurations')
        return trials

    def save_trial_result(self, trial: Trial, record: dict) -> None:
        """Persist Trial facts, update their scoring ledger, then refresh progress once."""
        if trial.job_directory != self.directory:
            raise ConfigError('Trial does not belong to this Job')
        verification_record = record.pop('verifier_result', None)
        invalidate = record['status'] == 'running' and self.verification_path.is_file()
        if invalidate or verification_record is not None:
            data = load_verification(self)
            if invalidate:
                data['trials'].pop(trial.id, None)
                data['configurations'].pop(f'{trial.execution.id}__attempt_{trial.attempt}', None)
            if verification_record is not None:
                data['trials'][trial.id] = verification_record
                record['verification_ref'] = self.build_verification_ref(f'/trials/{trial.id}')
            save_verification(self, data)
        trial.save_result(record)
        write_summary(self)

    def snapshot(self) -> dict:
        return {'schema_version': self.schema_version, 'job_id': self.id,
                'tasks': [convert_task_names(task.model_dump(mode='json'), legacy=self.legacy) for task in self.tasks],
                'executions': [execution.to_dict() for execution in self.executions],
                'n_attempts': self.n_attempts, 'concurrency': self.concurrency,
                'max_retries': self.max_retries}

    @classmethod
    def load(cls, directory: Path) -> Job:
        path = directory / 'job.json'
        if not path.is_file():
            raise ConfigError('Expected a Job directory containing job.json; legacy runs are unsupported')
        saved = json.loads(path.read_text())
        if saved.get('schema_version') not in {'1.0', '1.1'}:
            raise ConfigError('Unsupported Job snapshot; start a new Job')
        digest = saved.pop('identity')
        if identity(saved) != digest:
            raise ConfigError('Job snapshot identity mismatch')
        return cls(tasks=[LexVerseTask.model_validate(task) for task in saved['tasks']],
                   executions=[ExecutionConfig.from_dict(item, legacy=saved['schema_version'] == '1.0') for item in saved['executions']],
                   directory=directory, n_attempts=saved['n_attempts'], concurrency=saved['concurrency'],
                   max_retries=saved['max_retries'], id=saved['job_id'], schema_version=saved['schema_version'])


def load_trial(directory: Path, *, job_directory: Path | None = None) -> tuple[Job, Trial]:
    directory = directory.resolve()
    if job_directory is None:
        job_directory = next((parent for parent in directory.parents if (parent / 'job.json').is_file()), None)
        if job_directory is None:
            raise ConfigError('Expected a Trial inside a Job directory containing job.json; legacy runs are unsupported')
    job = Job.load(job_directory.expanduser().resolve())
    for trial in job.plan():
        if trial.directory == directory:
            return job, trial
    raise ConfigError('Trial does not belong to this Job')


def load_verification(job: Job) -> dict:
    path = job.verification_path
    return json.loads(path.read_text()) if path.exists() else {'configurations': {}, 'trials': {}}


def save_verification(job: Job, data: dict) -> None:
    data['status'] = compute_verification_status(job, data)
    atomic_write_json(job.verification_path, data)


def get_trial_verification_result(job: Job, trial: Trial, data: dict) -> dict:
    return data['trials'].get(trial.id, {'status': 'pending' if trial.task.verification else 'not_configured'})


def compute_verification_status(job, data, *, trials=None):
    planned = job.plan() if trials is None else trials
    states = [get_trial_verification_result(job, trial, data)['status'] for trial in planned if trial.task.verification]
    states.extend(item.get('status', 'pending') for item in data['configurations'].values())
    return ('not_configured' if not states else 'completed' if all(state == 'completed' for state in states)
            else 'pending' if all(state == 'pending' for state in states) else 'incomplete')


def summarize(job: Job) -> JobResult:
    """Derive progress from saved Trial and evaluation records without writing."""
    trials, groups = [], {}
    verification_records = load_verification(job)
    planned = job.plan()
    for trial in planned:
        path = trial.directory / 'result.json'
        record = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
        trials.append({'trial_id': trial.id, 'task_id': trial.task.id, 'config_id': trial.execution.id,
                       'attempt': trial.attempt, 'status': record['status'],
                       'result': str(path.relative_to(job.directory))})
        group = groups.setdefault(trial.execution.id, {'agent': trial.execution.agent,
                                  'model': trial.execution.model['name'], 'statuses': Counter()})
        group['statuses'][record['status']] += 1
    counts = Counter(trial['status'] for trial in trials)
    execution_status = ('completed' if counts['completed'] == len(trials) else
                        'running' if counts['running'] else 'needs_input' if counts['needs_input'] else
                        'pending' if counts['pending'] == len(trials) else 'incomplete')
    scoring_status = compute_verification_status(job, verification_records, trials=planned)
    summary: JobResult = {'schema_version': job.schema_version, 'job_id': job.id,
               'status': 'incomplete' if execution_status == 'completed' and scoring_status not in
                         {'completed', 'not_configured'} else execution_status,
               'planned_trials': len(trials), 'statuses': dict(counts), 'groups': groups,
               'execution_status': execution_status, 'verification_status': scoring_status, 'trials': trials}
    return summary


def write_summary(job: Job) -> JobResult:
    """Rebuild a small progress view; never use it as recovery state."""
    summary = summarize(job)
    atomic_write_json(job.directory / 'summary.json', convert_result_names(summary, legacy=job.legacy))
    return summary


def write_verification_report(job: Job) -> None:
    """Rebuild the report from authoritative execution and evaluation records."""
    from lexverse.benchmarks.aggregation import mean_metrics
    data = load_verification(job)
    rows = []
    trials = job.plan()
    records = {}
    for trial in trials:
        path = trial.directory / 'result.json'
        record = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
        records[trial.id] = {**record, 'verifier_result': get_trial_verification_result(job, trial, data)}
    for execution in job.executions:
        for attempt in range(1, job.n_attempts + 1):
            ledger = data['configurations'].get(f'{execution.id}__attempt_{attempt}', {})
            selected = [trial for trial in trials if trial.execution.id == execution.id and trial.attempt == attempt]
            for kind in dict.fromkeys(trial.task.source.task_type or trial.task.source.name for trial in selected):
                group = [trial for trial in selected if (trial.task.source.task_type or trial.task.source.name) == kind]
                scored = sum(records[trial.id]['verifier_result']['status'] == 'completed' for trial in group)
                statuses = Counter(records[trial.id]['status'] for trial in group)
                configured = any(trial.task.verification is not None for trial in group)
                failed = ledger.get('status') == 'failed' or statuses['failed'] or any(
                    records[trial.id]['verifier_result']['status'] == 'failed' for trial in group)
                status = ('failed' if failed else 'not_configured' if not configured
                          and statuses['completed'] == len(group) else
                          'completed' if scored == len(group) else 'incomplete')
                aggregate = ledger.get('aggregate', {}).get('tasks', {})
                if not scored:
                    metrics = {}
                elif kind in aggregate:
                    metrics = aggregate[kind]['metrics']
                else:
                    metrics = mean_metrics([dict(records[trial.id], task=kind) for trial in group
                                            if records[trial.id]['verifier_result']['status'] == 'completed'])[kind]['metrics']
                rows.append(dict(config_id=execution.id, agent=execution.agent, model=execution.model.get('name'),
                                 attempt=attempt, task=kind, status=status, total=len(group),
                                 completed=statuses['completed'], failed=statuses['failed'], scored=scored,
                                 metrics=metrics))
    root = job.verification_directory
    root.mkdir(exist_ok=True)
    fields = ['config_id', 'agent', 'model', 'attempt', 'task', 'status', 'total', 'completed', 'failed', 'scored']
    metrics = sorted({key for row in rows for key in row['metrics']})
    temporary = root / 'report.csv.tmp'
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ['metric.' + key for key in metrics])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields},
                             **{'metric.' + key: value for key, value in row['metrics'].items()}})
    temporary.replace(root / 'report.csv')


async def run_job(job: Job, execute: TrialExecutor, *, resume: bool = False,
                  prepare: Callable[[], Awaitable[None]] | None = None,
                  trial_id: str | None = None) -> dict:
    planned = job.plan()  # Validate everything before writing files or invoking an adapter.
    trials = [trial for trial in planned if trial.id == trial_id] if trial_id else planned
    if trial_id and not trials:
        raise ConfigError(f'Unknown Trial: {trial_id}')
    logger = logging.getLogger("lexverse.run")
    snapshot = job.snapshot()
    snapshot['identity'] = identity(snapshot)
    path = job.directory / 'job.json'
    if resume:
        if json.loads(path.read_text()) != snapshot:
            raise ConfigError('Job configuration changed; use a new directory')
    else:
        if job.directory.exists() and any(path.name != 'run.log' for path in job.directory.iterdir()):
            raise ConfigError('Job directory is not empty; use resume or a new directory')
        atomic_write_json(path, snapshot)

    write_summary(job)
    if prepare is not None:
        await prepare()

    logger.info("[run] executing total=%s concurrency=%s", len(trials), job.concurrency)

    async def worker(trial: Trial):
        return await trial.run(execute, job_id=job.id, resume=resume, max_retries=job.max_retries,
                               on_result=job.save_trial_result)

    outcomes = await Scheduler(job.concurrency).run(trials, worker)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return summarize(job)


def export_excel(job: Job) -> Path:
    """Optional per-Trial workbook, separate from the scoring comparison table."""
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise ConfigError("Excel export requires: pip install -e '.[reports]'") from exc
    book = Workbook()
    sheet = book.active
    sheet.title = 'Trials'
    sheet.append(['Trial', 'Task', 'Agent', 'Model', 'Config', 'Attempt', 'Status', 'Answer', 'Verification'])
    verification_records = load_verification(job)
    for trial in job.plan():
        record = json.loads((trial.directory / 'result.json').read_text())
        sheet.append([trial.id, trial.task.id, trial.execution.agent, trial.execution.model.get('name'),
                      trial.execution.id, trial.attempt, record['status'],
                      record.get('answer', record.get('response')), json.dumps(get_trial_verification_result(job, trial, verification_records), ensure_ascii=False)])
    path = job.directory / 'trials.xlsx'
    path.parent.mkdir(exist_ok=True)
    book.save(path)
    return path


_T = TypeVar("_T")
_R = TypeVar("_R")


class Scheduler(Generic[_T, _R]):
    def __init__(self, n_concurrent: int):
        if n_concurrent < 1:
            raise ValueError("n_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(n_concurrent)

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
            return await worker(item)
