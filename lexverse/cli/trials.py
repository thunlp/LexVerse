from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from lexverse.config import ConfigError


def summarize_atif_trajectory(path: Path) -> dict:
    from lexverse.models.trajectories import Trajectory

    traj = Trajectory.model_validate_json(path.read_text(encoding='utf-8'))

    tools = {}
    for step in traj.steps:
        for call in step.tool_calls or []:
            tools[call.function_name] = tools.get(call.function_name, 0) + 1

    return {'schema_version': traj.schema_version, 'total_steps': len(traj.steps),
            'total_tool_calls': sum(tools.values()), 'tool_calls_by_name': tools,
            'final_metrics': traj.final_metrics.model_dump(exclude_none=True) if traj.final_metrics else None}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='lexverse trial')
    commands = parser.add_subparsers(dest='command', required=True)

    inspect = commands.add_parser('inspect', help='read one Trial result')
    inspect.add_argument('trial_dir', type=Path)
    inspect.add_argument('--trajectory', action='store_true')

    resume = commands.add_parser('resume', help='resume one Trial')
    resume.add_argument('trial_dir', type=Path)
    mode = resume.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reply')
    mode.add_argument('--retry', action='store_true')
    resume.add_argument('--confirm-facts', action='store_true')

    args = parser.parse_args(argv)

    try:
        from lexverse.job import load_trial
        job, trial = load_trial(args.trial_dir.expanduser().resolve())
        record = json.loads((trial.directory / 'result.json').read_text())

        if args.command == 'inspect':
            if args.trajectory:
                path = trial.directory / 'trajectory.json'
                if not path.is_file() and record.get('trajectory_ref'):
                    candidate = Path(record['trajectory_ref'])
                    path = candidate if candidate.is_absolute() else trial.directory / candidate
                if path.resolve().is_relative_to(trial.directory.resolve()) and path.is_file():
                    try:
                        record['trajectory_summary'] = summarize_atif_trajectory(path)
                        record['trajectory_format'] = 'atif'
                    except ValueError:
                        record['trajectory_ref'] = str(path)
                        record['trajectory_format'] = 'native_or_unrecognized'
            print(json.dumps(record, ensure_ascii=False, indent=2))
            return 0

        if args.retry and args.confirm_facts:
            raise ConfigError('--confirm-facts requires --reply')
        if args.retry:
            from lexverse.job import load_verification
            missing_shared_verification = (
                record['status'] == 'completed' and trial.task.source.kind == 'user'
                and trial.task.verification is not None
                and trial.task.verification.config['environment_mode'] == 'shared'
                and load_verification(job)['trials'].get(trial.id, {}).get('status') != 'completed')
            if record['status'] not in {'failed', 'timeout', 'cancelled'} and not missing_shared_verification:
                raise ConfigError('Only a failed, timed-out or cancelled Trial can be retried')

            if trial.task.source.kind == 'user':
                from lexverse.tasks.runner import execute_trial
                result = asyncio.run(trial.run(lambda current: execute_trial(job, current),
                                               job_id=job.id, max_retries=0,
                                               on_result=job.save_trial_result))
                from lexverse.job import write_verification_report
                write_verification_report(job)
            else:
                from lexverse.benchmarks.runner import restore, _execute_job
                configs, plugin, bundle = restore(job)
                asyncio.run(_execute_job(job, configs, plugin, bundle, resume=True,
                                verify=False, trial_id=trial.id))
                result = json.loads((trial.directory / 'result.json').read_text())

            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'] in {'completed', 'needs_input'} else 4

        if record['status'] != 'needs_input' or trial.execution.agent != 'vanilla':
            raise ConfigError('Reply resume requires a paused vanilla Trial')

        from lexverse.tasks.runner import main as resume_agent
        return resume_agent({'args': {'command': 'resume', 'run_dir': str(trial.directory),
                                      'reply': args.reply, 'confirm_facts': args.confirm_facts}},
                            job_trial=(job, trial))

    except (ConfigError, OSError, ValueError) as exc:
        print(f'Trial error: {exc}', file=sys.stderr)
        return 2
