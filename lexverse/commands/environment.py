import json
from pathlib import Path
import subprocess
import sys
import tempfile


def setup(root=None):
    if sys.version_info < (3, 11):
        raise SystemExit("Stage2 requires Python 3.11 or newer")
    import tomllib
    from filelock import FileLock

    root = root or Path(__file__).resolve().parents[2]
    runtime = root / ".lexverse/runtime/stage2"
    requirements = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]["capability-task"]
    if root == Path(__file__).resolve().parents[2]:
        checked = subprocess.run([sys.executable, "-c",
            "from lexverse.agents.isolation import activate_dependencies; activate_dependencies(); "
            "import json, sys, importlib.metadata; from packaging.requirements import Requirement; "
            "requirements = [Requirement(value) for value in json.loads(sys.argv[1])]; "
            "assert all(importlib.metadata.version(req.name) in req.specifier for req in requirements)",
            json.dumps(requirements)],
            capture_output=True, text=True)
        if checked.returncode == 0:
            print(f"Stage2 dependencies reused: {runtime}")
            return 0
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime) + ".install.lock"), tempfile.TemporaryDirectory(dir=runtime.parent) as temporary:
        staged = Path(temporary) / "prepared"
        staged.mkdir()
        (staged / "site-packages").mkdir()
        report = staged / "resolution.json"
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(report), *requirements], check=True)
        plan = json.loads(report.read_text())["install"]
        lock = staged / "requirements.lock.txt"
        lock.write_text("\n".join(
            f"{item['metadata']['name']}=={item['metadata']['version']} --hash=sha256:{item['download_info']['archive_info']['hashes']['sha256']}"
            for item in plan
        ) + "\n")
        if plan:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes",
                "--target", str(staged / "site-packages"), "--requirement", str(lock),
            ], check=True)
        (staged / "runtime.json").write_text(json.dumps({
            "python": str(Path(sys.executable).resolve()),
            "python_version": list(sys.version_info[:2]),
            "packages": {item["metadata"]["name"]: item["metadata"]["version"] for item in plan},
        }, indent=2))
        previous = Path(temporary) / "previous"
        if runtime.exists():
            runtime.rename(previous)
        try:
            staged.rename(runtime)
        except OSError:
            if previous.exists():
                previous.rename(runtime)
            raise
    print(f"Stage2 dependencies prepared: {runtime}")


def build(sources, mode, retrieval_config=None):
    from lexverse.agents.isolation import activate_dependencies
    activate_dependencies()
    from tqdm import tqdm
    from lexverse.capabilities.knowledge.index import build_index
    from lexverse.tasks.user import RetrievalConfig
    root = Path(__file__).resolve().parents[2]
    cache = root / ".lexverse/capabilities"
    if retrieval_config is not None:
        import yaml
        config_path = retrieval_config.expanduser().resolve()
        try:
            config = RetrievalConfig.model_validate(yaml.safe_load(config_path.read_text()))
        except yaml.YAMLError as exc:
            raise ValueError("Invalid retrieval YAML") from exc
        config.embedding.resolve_path(config_path.parent)
        if mode is not None:
            config.mode = mode
    else:
        config = RetrievalConfig(mode=mode or "keyword")
    if config.overlap_tokens >= config.chunk_tokens:
        raise ValueError("retrieval overlap must be smaller than chunk size")
    embedding = None
    if config.mode != "keyword":
        from lexverse.capabilities.knowledge.embeddings import create_embeddings
        print("Loading local embedding model...", flush=True)
        embedding = create_embeddings(config.embedding, cache)
    for source in sources:
        print(f"Scanning source and checking cached index: {source}", flush=True)
        with tqdm(desc="Index", unit="file", mininterval=1) as bar:
            def progress(completed, total, chunks):
                bar.total = total
                bar.set_postfix(chunks=chunks, refresh=False)
                bar.update(completed - bar.n)
            path = build_index(source.expanduser().resolve(), cache, config, embedding, progress=progress)
        manifest = json.loads((path / "manifest.json").read_text())
        print(f"Index ready: {path}; chunks={manifest['chunk_count']}", flush=True)
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="lexverse environment")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="install or reuse isolated Stage2 dependencies")
    index = commands.add_parser("build", help="build or reuse knowledge indexes")
    index.add_argument("--knowledge", type=Path, nargs="+", required=True)
    index.add_argument("--mode", choices=["keyword", "vector", "hybrid"],
                       help="retrieval mode (default: config value, otherwise keyword)")
    index.add_argument("--retrieval-config", type=Path, help="YAML containing a retrieval configuration")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            return setup() or 0
        if args.retrieval_config is not None:
            return build(args.knowledge, args.mode, args.retrieval_config)
        return build(args.knowledge, args.mode)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"environment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
