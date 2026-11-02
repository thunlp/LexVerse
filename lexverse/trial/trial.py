from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Awaitable, Callable

from lexverse.utils.files import atomic_write_json
from lexverse.models.verification_compat import convert_result_names
from lexverse.models.task.task import LexVerseTask
from lexverse.models.trial.config import ExecutionConfig
from lexverse.trial.errors import TrialError, TrialTimeoutError


@dataclass(frozen=True)
class Trial:
    id: str
    task: LexVerseTask
    execution: ExecutionConfig
    attempt: int
    directory: Path
    retry: int = 0
    job_directory: Path | None = None

    @property
    def legacy(self):
        return getattr(self.execution, "_legacy", False)

    async def run(self, execute: TrialExecutor, *, job_id: str, resume: bool = False,
                  max_retries: int = 0,
                  on_result: Callable[[Trial, dict], None] | None = None) -> dict:
        """Execute or resume one Trial, preserving its identity across retries."""
        logger = logging.getLogger("lexverse.run")
        result_path = self.directory / 'result.json'

        def save(record):
            if on_result is None:
                self.save_result(record)
            else:
                on_result(self, record)

        # Resume: reuse paused results or completed results with valid artifacts.
        if resume and result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior['status'] == 'needs_input':
                return convert_result_names(prior)
            if prior['status'] == 'completed' and self._completed_artifacts_valid(prior):
                return convert_result_names(prior)

        # Prepare: create result directories and establish the Trial identity.
        internal = self.directory / '.internal'
        internal.mkdir(parents=True, exist_ok=True)
        (self.directory / 'artifacts').mkdir(exist_ok=True)
        base = self.identity_record(job_id)
        prior_retry = json.loads(result_path.read_text()).get('retry', -1) if result_path.exists() else -1

        for retry in range(prior_retry + 1, prior_retry + max_retries + 2):
            # History: preserve the previous result before starting another retry.
            if result_path.exists() and retry > 0:
                history = internal / 'result_history' / f'retry-{retry - 1}.json'
                history.parent.mkdir(exist_ok=True)
                if not history.exists():
                    shutil.copy2(result_path, history)

            # Start: persist the running state and emit the lifecycle event.
            record = {**base, 'status': 'running', 'retry': retry}
            save(record)
            _append_event(self.directory, 'started', retry=retry)
            logger.info('[task] start task=%s config=%s attempt=%s retry=%s', self.task.id, self.execution.id, self.attempt, retry)

            try:
                # Execute: give the adapter isolated task/config copies, including mutable world state.
                isolated = Trial(self.id, self.task.model_copy(deep=True),
                                 ExecutionConfig.from_dict(json.loads(json.dumps(self.execution.to_dict())),
                                                           legacy=getattr(self.execution, "_legacy", False)),
                                 self.attempt, self.directory, retry, self.job_directory)
                result = await execute(isolated)
                retryable = result.pop('_retryable', False)
                record.update(result)
                record.update(base, retry=retry)
                if retryable:
                    raise TrialError(result.get('error_message', 'Trial execution failed'))
            except asyncio.CancelledError:
                # Cancellation: save partial results and propagate the cancellation.
                self._trajectory_result(record, retry)
                record.update(status='cancelled')
                save(record)
                _append_event(self.directory, 'cancelled', retry=retry)
                raise
            except Exception as exc:
                # Failure: record the error and retry eligible failures within the budget.
                record.update(status='timeout' if isinstance(exc, (TimeoutError, TrialTimeoutError)) else 'failed',
                              error_type=type(exc).__name__, error=str(exc))
                if isinstance(exc, TrialError) and retry < prior_retry + max_retries + 1:
                    self._trajectory_result(record, retry)
                    save(record)
                    _append_event(self.directory, 'retry', retry=retry, error_type=type(exc).__name__)
                    continue

            # Finish: persist the outcome and return it to the Job scheduler.
            self._trajectory_result(record, retry)
            save(record)
            _append_event(self.directory, record['status'], retry=retry)
            logger.info('[task] %s task=%s config=%s attempt=%s', record['status'], self.task.id, self.execution.id, self.attempt)
            return record

    def identity_record(self, job_id):
        return {'trial_id': self.id, 'task_id': self.task.id, 'config_id': self.execution.id,
                'id': self.task.id, 'task': self.task.source.task_type or self.task.source.name,
                'source': self.task.source.model_dump(mode='json'), 'sample_id': self.task.source.sample_id,
                'attempt': self.attempt, 'job_id': job_id,
                **({'verification_ref': f"{'evaluation/evaluation' if self.legacy else 'verification/verification'}.json#/trials/{self.id}"} if self.task.verification else {})}

    def _completed_artifacts_valid(self, record):
        if record.get('outputs', {}).get('status') == 'missing':
            return False
        for item in self.task.outputs.get('artifacts', []):
            if item.get('required', True) and not (self.directory / 'artifacts' / item['destination']).exists():
                return False
        artifacts = record.get('execution_artifacts', record.get('artifacts', {}))
        if isinstance(artifacts, dict):
            return all(Path(path).exists() for path in artifacts.values())
        for item in artifacts:
            path = Path(item['path'])
            if not path.is_file():
                return False
            if item.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                return False
        return True

    def save_result(self, record):
        atomic_write_json(self.directory / 'result.json', convert_result_names(record, legacy=self.legacy))
        if record['status'] not in {'running', 'needs_input'}:
            from lexverse.models.trial.paths import RunPaths
            run = (self.directory / '.internal/native' / f"retry-{record.get('retry', 0)}" / 'agent'
                   if self.task.source.kind == 'benchmark' else self.directory)
            RunPaths(run).worker_result.unlink(missing_ok=True)

    def _trajectory_result(self, record, retry):
        if self.execution.agent in ('static', 'native'):
            return
        from lexverse.models.trial.paths import RunPaths
        root = (self.directory / '.internal/native' / f'retry-{retry}' / 'agent'
                if self.task.source.kind == 'benchmark' else self.directory)
        worker_result = RunPaths(root).worker_result
        if worker_result.is_file():
            saved = json.loads(worker_result.read_text())
            for key in ('trajectory_status', 'trajectory_error', 'usage'):
                if key in saved:
                    record[key] = saved[key]
            if 'artifacts' not in record:
                artifacts = saved.get('artifacts', [])
                record['artifacts'] = (artifacts if self.task.source.kind == 'user' else
                                       {item['name']: item['path'] for item in artifacts})
        if record.get('trajectory_status') == 'saved' and (self.directory / 'trajectory.json').is_file():
            record['trajectory_ref'] = str(self.directory / 'trajectory.json')



TrialExecutor = Callable[[Trial], Awaitable[dict]]


def _append_event(directory: Path, event: str, **details) -> None:
    path = directory / '.internal/lifecycle.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'event': event, 'time': time.time(), **details}) + '\n')
