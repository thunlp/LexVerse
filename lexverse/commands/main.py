import argparse
import sys


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog="lexverse", description="LexVerse benchmark and task environment")
    groups = parser.add_subparsers(dest="group", required=True)

    for name, help_text in (("benchmark", "run and evaluate benchmarks"),
                            ("task", "run and resume user tasks"),
                            ("environment", "prepare dependencies and knowledge indexes")):
        groups.add_parser(name, help=help_text, add_help=False)

    args, remaining = parser.parse_known_args(arguments)

    if args.group == "task":
        from lexverse.commands.tasks import main as task_main
        return task_main(remaining)

    if args.group == "environment":
        from lexverse.commands.environment import main as environment_main
        return environment_main(remaining)

    if args.group == "benchmark":
        from lexverse.commands.benchmarks import main as benchmark_main
        return benchmark_main(remaining)

    parser.error(f"Unknown command group: {args.group}")

if __name__ == "__main__":
    raise SystemExit(main())
