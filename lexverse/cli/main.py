import argparse
import sys


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog="lexverse", description="LexVerse benchmark and task environment")
    groups = parser.add_subparsers(dest="group", required=True)

    for name, help_text in (("run", "start a Job"),
                            ("task", "manage task definitions"),
                            ("job", "manage Jobs"),
                            ("trial", "inspect and resume a Trial"),
                            ("agents", "list Agent choices"),
                            ("resources", "download and build resources"),
                            ("setup", "prepare Agent dependencies")):
        groups.add_parser(name, help=help_text, add_help=False)

    args, remaining = parser.parse_known_args(arguments)

    if args.group == "run":
        from lexverse.cli.jobs import run_main
        return run_main(remaining)

    if args.group == "job":
        from lexverse.cli.jobs import main as job_main
        return job_main(remaining)

    if args.group == "task":
        from lexverse.cli.tasks import main as task_main
        return task_main(remaining)

    if args.group == "trial":
        from lexverse.cli.trials import main as trial_main
        return trial_main(remaining)

    if args.group == "agents":
        from lexverse.cli.agents import main as agents_main
        return agents_main(remaining)

    if args.group == "resources":
        from lexverse.cli.resources import main as resources_main
        return resources_main(remaining)

    if args.group == "setup":
        from lexverse.cli.resources import setup_main
        return setup_main(remaining)

    parser.error(f"Unknown command group: {args.group}")

if __name__ == "__main__":
    raise SystemExit(main())
