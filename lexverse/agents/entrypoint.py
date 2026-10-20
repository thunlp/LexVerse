from lexverse.agents.isolation import activate_dependencies
import sys


def main():
    try:
        activate_dependencies()
    except RuntimeError as exc:
        # Allow external general CLI agents (`--agent claude-code`, `--agent oracle`, etc.),
        # `task inspect`, and `task agents` to run even when the local agent LangGraph
        # dependency bundle has not been installed via `lexverse environment setup`.
        args = sys.argv[1:]
        needs_agent_dependencies = True
        if args and args[0] in ("init", "inspect", "agents", "benchmark", "-h", "--help"):
            needs_agent_dependencies = False
        elif "--agent" in args:
            idx = args.index("--agent")
            if idx + 1 < len(args) and args[idx + 1] != "lexverse":
                needs_agent_dependencies = False
        if needs_agent_dependencies:
            print(f"Task setup failed: {exc}", file=sys.stderr)
            return 2
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from lexverse.agents.benchmark_runner import main as benchmark_main
        return benchmark_main(sys.argv[2:])
    from lexverse.commands.tasks import main as user_main
    return user_main()


if __name__ == "__main__":
    raise SystemExit(main())
