from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lexverse.config import ConfigError, SecretsMissing, j1_summary_default
from lexverse.agents.factory import AgentsSelection, validate_agent
from lexverse.models.job.config import JobConfig
from lexverse.cli.config import read_job_config, write_job_config


def add_execution_arguments(parser):
    parser.add_argument('-c', '--config', type=Path,
                        help='read only this Job YAML; skip automatic local/example lookup')

    source = parser.add_mutually_exclusive_group()
    source.add_argument('-p', '--path', type=Path,
                        help='Task package; look up YAML by task.toml ID, or check the -c source matches')
    source.add_argument('-b', '--benchmark',
                        help='benchmark name; look up YAML by name, or check the -c source matches')

    parser.add_argument('--model', help='override the tested model for all Agent entries')
    parser.add_argument('--profile', help='override the default model secrets profile')
    parser.add_argument('--agents', action=AgentsSelection,
                        help='replace Agent entries with these comma-separated names')

    parser.add_argument('--environment', choices=('docker', 'local'))
    parser.add_argument('--attempts', type=int)
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--max-retries', type=int)

    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--export-excel', action=argparse.BooleanOptionalAction, default=None)
    parser.epilog = (
        'Without -c, search ./configs/jobs/<name>.local.yaml, then <name>.yaml (example). '
        'The name is the benchmark name or task.toml ID. Read one file only; invalid files '
        'do not fall back. CLI runtime options override YAML; -b/-p must match its source '
        'or supply a missing source. With no YAML, source, --model and --agents are required. '
        'YAML paths are relative to its directory; CLI paths are relative to the current directory.'
    )


def _task_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path.parent if path.name == 'task.toml' else path


def _select_config(args):
    if args.config is not None:
        return args.config.expanduser().resolve(), []
    name = args.benchmark
    if args.path is not None:
        from lexverse.tasks.user import load_user_task
        name = load_user_task(args.path).id
    if name is None:
        return None, []
    # A benchmark name, like a Task ID, must be a single filename component.
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ConfigError('Benchmark name must be a single name, not a path')
    directory = Path.cwd() / 'configs/jobs'
    candidates = [directory / f'{name}.local.yaml', directory / f'{name}.yaml']
    for path in candidates:
        if path.exists() or path.is_symlink():
            return path.resolve(), candidates
    return None, candidates


def _check_source(data, args, base: Path):
    if args.path is not None:
        requested = _task_directory(args.path)
        saved = data.get('path')
        if saved is not None and not isinstance(saved, (str, Path)):
            raise ConfigError('YAML path must be a path string')
        if (data.get('benchmark') is not None or (saved is not None and
                _task_directory(base / Path(saved).expanduser()) != requested)):
            raise ConfigError(f'Source conflict: CLI Task {requested}; YAML path={saved}, '
                              f'benchmark={data.get("benchmark")}')
        data['path'] = requested
    if args.benchmark is not None:
        saved = data.get('benchmark')
        if data.get('path') is not None or (saved is not None and
                (not isinstance(saved, dict) or saved.get('name') != args.benchmark)):
            raise ConfigError(f'Source conflict: CLI benchmark {args.benchmark}; '
                              f'YAML path={data.get("path")}, benchmark={saved}')
        if saved is None:
            data['benchmark'] = {'name': args.benchmark}


def config_from_args(args) -> JobConfig:
    config_path, candidates = _select_config(args)
    args.loaded_config = config_path
    if config_path is not None:
        data = read_job_config(config_path)
        try:
            JobConfig.reject_inline_credentials(data)
            _check_source(data, args, config_path.parent)
            data = JobConfig.model_validate(data).resolve(config_path.parent).model_dump(mode='python')
        except ValueError as exc:
            raise ConfigError(f'Invalid Job configuration {config_path}: {exc}') from exc
    else:
        missing = [name for name, value in (('-b/--benchmark or -p/--path', args.path or args.benchmark),
                   ('--model', args.model), ('--agents', args.agents)) if not value]
        if missing:
            searched = '; searched: ' + ', '.join(map(str, candidates)) if candidates else ''
            raise ConfigError('No Job YAML loaded; missing ' + ', '.join(missing) + searched)
        data = {'models': {'default': {'provider': 'openai_compatible', 'name': args.model,
                                      'profile': args.profile}},
                'agents': [{'name': name} for name in args.agents]}
        _check_source(data, args, Path.cwd())

    if args.model is not None:
        data['models']['default']['name'] = args.model
        for agent in data['agents']:
            if isinstance(agent, dict):
                agent['model'] = None
    if args.profile is not None:
        data['models']['default']['profile'] = args.profile

    if args.agents is not None:
        data['agents'] = [{'name': name} for name in args.agents]

    for name in ('attempts', 'concurrency', 'max_retries', 'output_dir', 'interactive', 'export_excel'):
        value = getattr(args, name)
        if value is not None:
            data[name] = value

    if args.environment is not None:
        data['environment'] = {**data.get('environment', {}), 'type': args.environment}

    config = JobConfig.model_validate(data).resolve(Path.cwd())

    if config.benchmark:
        from lexverse.benchmarks.registry import get_plugin
        if config.benchmark['name'] == 'j1bench':
            config.models.setdefault('summary', j1_summary_default(config.models['default'].get('profile')))
        supported = get_plugin(config.benchmark['name']).supported_agents
        source = config.benchmark['name']
        network_mode = 'public'
    else:
        from lexverse.tasks.user import SUPPORTED_AGENTS, load_user_task
        supported, source = SUPPORTED_AGENTS, 'user'
        task = load_user_task(config.path)
        network_mode = task.environment.network_mode
        if (config.environment.type == 'local' and task.verifier
                and task.verifier.environment_mode == 'separate' and not getattr(args, 'no_verify', False)):
            raise ConfigError('Local cannot provide separate verification; use Docker or shared verification')

    for agent in config.agents:
        validate_agent(agent.name, supported, source=source, model=agent.model or config.models['default'],
                       environment=config.environment.type, network_mode=network_mode)

    return config


def run_main(argv=None):
    parser = argparse.ArgumentParser(prog='lexverse run')
    add_execution_arguments(parser)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('--no-verify', action='store_true')

    # Accept historical CLI spellings without advertising them as new commands.
    argv = list(sys.argv[2:] if argv is None else argv)
    argv = ['--no-verify' if item == '--no-evaluate' else item for item in argv]
    if argv and argv[0] == 'evaluate':
        argv[0] = 'verify'
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        print(f'Job configuration: {args.loaded_config or "CLI (no YAML found)"}', file=sys.stderr)

        if args.print_config:
            print(json.dumps(config.model_dump(mode='json', exclude_none=True), ensure_ascii=False, indent=2))
            return 0

        for agent in config.agents:
            model = agent.model or config.models['default']
            print(f'Agent: {agent.name}; model: {model.get("name")}; '
                  f'profile: {model.get("profile") or "(none)"}', file=sys.stderr)

        if config.benchmark:
            from lexverse.benchmarks.runner import command
            return command(config, verify=not args.no_verify, dry_run=args.dry_run)

        from lexverse.tasks.runner import command
        return command(config, verify=not args.no_verify, dry_run=args.dry_run)

    except (ConfigError, ValueError, OSError, SecretsMissing) as exc:
        print(f'config error: {exc}', file=sys.stderr)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(prog='lexverse job')
    commands = parser.add_subparsers(dest='command', required=True)

    initialize = commands.add_parser('init', help='save an experiment without running it')
    initialize.add_argument('destination', type=Path)
    add_execution_arguments(initialize)

    start = commands.add_parser('start', help='start a Job')
    add_execution_arguments(start)
    start.add_argument('--dry-run', action='store_true')
    start.add_argument('--print-config', action='store_true')
    start.add_argument('--no-verify', action='store_true')

    commands.add_parser('schema', help='print Job JSON Schema')

    inspect = commands.add_parser('inspect', help='read Job status')
    inspect.add_argument('job_dir', type=Path)

    resume = commands.add_parser('resume', help='resume a Job')
    resume.add_argument('job_dir', type=Path)
    resume.add_argument('--no-verify', action='store_true')

    verify = commands.add_parser('verify', help='score saved answers')
    verify.add_argument('job_dir', type=Path)

    report = commands.add_parser('report', help='rebuild derived reports')
    report.add_argument('job_dir', type=Path)
    report.add_argument('--export-excel', action='store_true')

    # Accept historical CLI spellings without advertising them as new commands.
    argv = list(sys.argv[2:] if argv is None else argv)
    argv = ['--no-verify' if item == '--no-evaluate' else item for item in argv]
    if argv and argv[0] == 'evaluate':
        argv[0] = 'verify'
    args = parser.parse_args(argv)

    try:
        if args.command == 'start':
            return run_main(argv[1:] if argv is not None else sys.argv[3:])

        if args.command == 'schema':
            print(json.dumps(JobConfig.model_json_schema(), ensure_ascii=False, indent=2))
            return 0

        if args.command == 'init':
            config = config_from_args(args)
            write_job_config(config, args.destination)
            print(json.dumps({'config': str(args.destination.expanduser().absolute())}))
            return 0

        if args.command == 'inspect':
            from lexverse.job import Job, summarize
            print(json.dumps(summarize(Job.load(args.job_dir.expanduser().resolve())), ensure_ascii=False, indent=2))
            return 0

        if args.command == 'resume':
            from lexverse.job import Job
            job = Job.load(args.job_dir.expanduser().resolve())
            if job.tasks[0].source.kind == 'user':
                from lexverse.tasks.runner import _resume_job
            else:
                from lexverse.benchmarks.runner import _resume_job
            return _resume_job(job, verify=not args.no_verify)

        if args.command in {'verify', 'evaluate'}:
            from lexverse.job import Job
            job = Job.load(args.job_dir.expanduser().resolve())
            if job.tasks[0].source.kind == 'user':
                from lexverse.tasks.runner import _verify_job
            else:
                from lexverse.benchmarks.runner import _verify_saved_job as _verify_job
            return _verify_job(job)

        from lexverse.job import Job, write_summary, write_verification_report, export_excel
        job = Job.load(args.job_dir.resolve())
        summary = write_summary(job)
        write_verification_report(job)
        if args.export_excel:
            export_excel(job)
        return 0 if summary['status'] == 'completed' else 4

    except (ConfigError, ValueError, OSError, SecretsMissing) as exc:
        print(f'config error: {exc}', file=sys.stderr)
        return 2
