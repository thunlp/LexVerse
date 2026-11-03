from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lexverse.config import ConfigError
from lexverse.models.task.config import UserTaskSpec


def init_user_task(path: Path) -> dict:
    """Create a task package without choosing a model or Agent."""
    path = path.expanduser().absolute()

    if path.exists() or path.is_symlink():
        raise ConfigError(f'Task path already exists: {path}')

    UserTaskSpec(id=path.name)

    files = {
        'task.toml': f'''schema_version = "1.0"
id = {json.dumps(path.name)}

[[artifacts]]
source = "/work/report.md"
destination = "report.md"

[environment]

[verifier]
environment_mode = "separate"
''',
        'instruction.md': 'Write a report to /work/report.md. Replace this instruction with your task.\n\nTask materials, if provided, are available in /inputs/.\n',
        'environment/Dockerfile': 'FROM python:3.12-slim\nCOPY inputs/ /inputs/\nWORKDIR /work\n',
        'tests/Dockerfile': 'FROM python:3.12-slim\nCOPY . /tests/\n',
        'tests/test.sh': '#!/bin/sh\nset -eu\npython /tests/grade.py\n',
        'tests/grade.py': '''# Replace this smoke check with the task's scoring criteria.
import json
from pathlib import Path
report = Path("/work/report.md")
reward = float(report.is_file() and bool(report.read_text().strip()))
output = Path("/logs/verifier")
output.mkdir(parents=True, exist_ok=True)
(output / "reward.json").write_text(json.dumps({"reward": reward}))
''',
    }

    path.mkdir(parents=True)
    (path / 'environment/inputs').mkdir(parents=True)
    for name, content in files.items():
        destination = path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding='utf-8')

    return {'task_dir': str(path), 'created': [str(path / name) for name in files]}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='lexverse task')
    commands = parser.add_subparsers(dest='command', required=True)

    initialize = commands.add_parser('init', help='create a TOML task package')
    initialize.add_argument('path', type=Path)

    catalog = commands.add_parser('catalog', help='inspect a benchmark task catalog')
    from lexverse.benchmarks.registry import available_plugins
    catalog.add_argument('benchmark', choices=available_plugins())

    args = parser.parse_args(argv)

    try:
        if args.command == 'init':
            payload = init_user_task(args.path)
        else:
            from lexverse.benchmarks.registry import get_plugin
            plugin = get_plugin(args.benchmark)
            payload = {'benchmark': plugin.name, 'source_version': plugin.source_version,
                       'default_agent': plugin.default_agent,
                       'supported_agents': sorted(plugin.supported_agents),
                       'tasks': plugin.load_catalog()['tasks']}

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    except (ConfigError, OSError, ValueError) as exc:
        print(f'Task error: {exc}', file=sys.stderr)
        return 2
