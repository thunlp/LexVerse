import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from lexverse.runtime.paths import AGENT_RUNTIME_ROOT, CAPABILITIES_ROOT, PROJECT_ROOT
from lexverse.runtime.dependencies import is_macos_arm64


def setup(root=None):
    if sys.version_info < (3, 12):
        raise SystemExit("Agent dependencies require Python 3.12 or newer")
    import tomllib
    from filelock import FileLock

    root = root or PROJECT_ROOT
    runtime = root / AGENT_RUNTIME_ROOT.relative_to(PROJECT_ROOT)
    config = tomllib.loads((root / "pyproject.toml").read_text())
    project = config["project"]
    requirements = project["optional-dependencies"]["capability-task"]
    macos = is_macos_arm64()
    if macos and (not (Path(sys.prefix) / "conda-meta").is_dir()
                  or ".".join(map(str, sys.version_info[:2])) != config["tool"]["lexverse"]["macos-arm64"]["python-version"]):
        raise RuntimeError("Activate the macOS arm64 Conda environment described in README.md before setup")
    if macos:
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda:
            raise RuntimeError("Activate the macOS arm64 Conda environment described in README.md before setup")
        subprocess.run([
            conda, "install", "--yes", "--prefix", sys.prefix,
            "--override-channels", "--channel", "conda-forge", "--strict-channel-priority",
            *config["tool"]["lexverse"]["macos-arm64"]["conda-packages"],
        ], env={**os.environ, "CONDA_SUBDIR": "osx-arm64"}, check=True)
    if root == PROJECT_ROOT:
        checked = subprocess.run([sys.executable, "-c",
            "from lexverse.runtime.dependencies import activate_dependencies; activate_dependencies(); "
            "import json, sys, importlib.metadata; from packaging.requirements import Requirement; "
            "requirements = [Requirement(value) for value in json.loads(sys.argv[1])]; "
            "assert all(importlib.metadata.version(req.name) in req.specifier for req in requirements "
            "if req.marker is None or req.marker.evaluate())",
            json.dumps(requirements)],
            capture_output=True, text=True)
        if checked.returncode == 0:
            print(f"Agent dependencies reused: {runtime}")
            return 0
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime) + ".install.lock"), tempfile.TemporaryDirectory(dir=runtime.parent) as temporary:
        staged = Path(temporary) / "prepared"
        staged.mkdir()
        (staged / "site-packages").mkdir()
        report = staged / "resolution.json"
        constraints = []
        if macos:
            constraint = staged / "constraints.txt"
            constraint.write_text("\n".join(config["tool"]["lexverse"]["macos-arm64"]["pip-constraints"]) + "\n")
            constraints = ["--constraint", str(constraint)]
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(report),
                        *constraints, *requirements], check=True)
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
            **({"environment": "conda"} if macos else {}),
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
    print(f"Agent dependencies prepared: {runtime}")


def build(sources, mode, retrieval_config=None):
    from lexverse.runtime.dependencies import activate_dependencies
    activate_dependencies()
    from tqdm import tqdm
    from lexverse.capabilities.knowledge.index import build_index
    from lexverse.capabilities.knowledge.download import register_collection
    from lexverse.tasks.user import RetrievalConfig
    cache = CAPABILITIES_ROOT
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
        record = register_collection(source.expanduser(), path, config, cache=cache)
        print(f"Index ready: {path}; chunks={record['chunk_count']}", flush=True)
        print(f"Collection registered: {record['id']}", flush=True)
    return 0


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="lexverse environment")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="install or reuse isolated agent dependencies")
    download = commands.add_parser("download", help="download knowledge or skill resources")
    resources = download.add_subparsers(dest="resource", required=True)
    knowledge = resources.add_parser("knowledge", help="download upstream and LexVerse-hosted knowledge")
    from lexverse.capabilities.knowledge.download import DEFAULT_ROOT, add_arguments
    add_arguments(knowledge)
    indexes = resources.add_parser("indexes", help="download and verify prebuilt knowledge collections from HF")
    indexes.add_argument("--collections", nargs="+", help="Collection IDs (default: all published collections)")
    indexes.add_argument("--local-dir", type=Path, default=DEFAULT_ROOT, help="Capability cache root")
    indexes.add_argument("--revision", default="main", help="HF dataset revision (default: main)")
    indexes.add_argument("--dry-run", action="store_true", help="Fetch the catalog and list selected indexes without downloading them")
    skills = resources.add_parser("skills", help="download or reuse registered skill sources")
    skills.add_argument("--sources", nargs="+", help="Download only the named skill source repositories")
    skills.add_argument("--manifest", type=Path, help="Local download YAML (default: fetch skills/sources.yaml from LexVerse HF)")
    skills.add_argument("--local-dir", type=Path, default=DEFAULT_ROOT, help="Capability cache root (default: <LexVerse root>/.lexverse/capabilities)")
    skills.add_argument("--dry-run", action="store_true", help="List skills without downloading source archives (fetches the HF manifest unless --manifest is supplied)")
    index = commands.add_parser("build", help="build or reuse knowledge indexes")
    index.add_argument("--knowledge", type=Path, nargs="+", required=True)
    index.add_argument("--mode", choices=["keyword", "vector", "hybrid"],
                       help="retrieval mode (default: config value, otherwise keyword)")
    index.add_argument("--retrieval-config", type=Path, help="YAML containing a retrieval configuration")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            return setup() or 0
        if args.command == "download":
            if args.resource == "indexes":
                from lexverse.capabilities.knowledge.download import download_indexes
                return download_indexes(args)
            if args.resource == "knowledge":
                from lexverse.capabilities.knowledge.download import run
                return run(args)
            from lexverse.capabilities.skills import download_skills, load_download_registry
            cache = args.local_dir.resolve()
            registry = load_download_registry(args.manifest, cache_dir=str(cache / "cache" / "huggingface"))
            return download_skills(registry, args.sources, cache, dry_run=args.dry_run)
        if args.retrieval_config is not None:
            return build(args.knowledge, args.mode, args.retrieval_config)
        return build(args.knowledge, args.mode)
    except (RuntimeError, ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"environment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
