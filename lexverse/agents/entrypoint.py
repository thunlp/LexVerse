from lexverse.agents.isolation import activate_dependencies
import sys


def main():
    try:
        activate_dependencies()
    except RuntimeError as exc:
        print(f"Task setup failed: {exc}", file=sys.stderr)
        return 2
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from lexverse.agents.benchmark_runner import main as benchmark_main
        return benchmark_main(sys.argv[2:])
    from lexverse.commands.tasks import main as user_main
    return user_main()


if __name__ == "__main__":
    raise SystemExit(main())
