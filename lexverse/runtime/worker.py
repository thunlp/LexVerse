import argparse
import json
from pathlib import Path
import sys

from lexverse.runtime.dependencies import activate_dependencies


def main(argv=None):
    # Parse the worker kind and request path.
    parser = argparse.ArgumentParser(prog="lexverse runtime worker")
    parser.add_argument("kind", choices=("task", "benchmark"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)

    # Load the request prepared by the parent process.
    payload = json.loads(args.request.read_text(encoding="utf-8"))

    # Activate dependencies before importing execution modules.
    if payload["dependencies_required"]:
        try:
            activate_dependencies()
        except RuntimeError as exc:
            message = f"{args.kind} requires Agent capability dependencies: {exc}"
            print(message, file=sys.stderr)
            if payload.get("reply_path"):
                Path(payload["reply_path"]).write_text(json.dumps({"error": message}), encoding="utf-8")
            return 2

    # Dispatch to the task or benchmark enhancement entry point.
    if args.kind == "task":
        from lexverse.runtime.user_task import main as execute
    else:
        from lexverse.runtime.enhancement import run_benchmark as execute
    return execute(payload)


if __name__ == "__main__":
    raise SystemExit(main())
