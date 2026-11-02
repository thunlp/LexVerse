import argparse
import json
from lexverse.agents.factory import AgentFactory
from lexverse.benchmarks.registry import available_plugins, get_plugin


def main(argv=None):
    parser = argparse.ArgumentParser(prog='lexverse agents')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='list registered Agents')
    parser.parse_args(argv)
    from lexverse.tasks.user import SUPPORTED_AGENTS
    print(json.dumps({'built_in': ['static', 'native', 'vanilla'],
                      'external': [name for name in AgentFactory.supported_agent_names()
                                   if name != 'vanilla'],
                      'sources': {'user': sorted(SUPPORTED_AGENTS),
                                  **{name: sorted(get_plugin(name).supported_agents) for name in available_plugins()}}},
                     ensure_ascii=False, indent=2))
    return 0
