import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from filelock import FileLock
import yaml

from lexverse.config import ConfigError
from lexverse.runtime.paths import CAPABILITIES_ROOT as DEFAULT_ROOT

DATASET_REPO = "TsinghuaNLP/LexVerse"
CATALOG = "indexes/collections.json"
RETRIEVAL = "indexes/retrieval.yaml"


def download_plan(manifest, local_dir, sources=None, lexverse_revision="main"):
    if not isinstance(manifest, dict):
        raise ValueError("The download manifest must be a mapping of source names")
    root = local_dir.resolve()
    selected = list(dict.fromkeys(sources)) if sources else list(manifest)
    plan = []
    for name in selected:
        spec = manifest.get(name)
        if not isinstance(spec, dict) or spec.get("delivery") not in {"upstream", "lexverse"}:
            raise ValueError(f"Unknown source or delivery type: {name}")
        if spec["delivery"] == "lexverse":
            for item in spec["files"]:
                local = item["local_path"]
                target = local_target(root, local)
                path = quote(local, safe="/")
                url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{lexverse_revision}/{path}"
                plan.append((name, url, target, True))
            continue
        revision = spec["upstream_revision"]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"A full upstream commit SHA is required: {name}")
        repo = urlparse(spec["upstream_url"])
        if repo.scheme != "https" or repo.netloc not in {"github.com", "huggingface.co"}:
            raise ValueError(f"Unsupported upstream repository: {name}")
        for item in spec["files"]:
            remote = PurePosixPath(item["upstream_path"])
            target = local_target(root, item["local_path"])
            if remote.is_absolute() or ".." in remote.parts:
                raise ValueError(f"Invalid file path: {name}")
            path = quote(remote.as_posix(), safe="/")
            if repo.netloc == "huggingface.co":
                if not repo.path.startswith("/datasets/"):
                    raise ValueError(f"Expected a Hugging Face dataset repository: {name}")
                repo_id = repo.path.removeprefix("/datasets/").rstrip("/")
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"
            else:
                repo_id = repo.path.strip("/")
                host = "media.githubusercontent.com/media" if spec.get("git_lfs") else "raw.githubusercontent.com"
                url = f"https://{host}/{repo_id}/{revision}/{path}"
            plan.append((name, url, target, repo.netloc == "huggingface.co"))
    return plan


def local_target(root, local):
    path = PurePosixPath(local)
    target = (root / local).resolve()
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "knowledge" or not target.is_relative_to(root):
        raise ValueError(f"Invalid local knowledge path: {local}")
    return target


def load_manifest(path=None, *, cache_dir=None):
    revision = "main"
    if path is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        try:
            path = Path(hf_hub_download(DATASET_REPO, "knowledge/sources.yaml", repo_type="dataset", cache_dir=cache_dir))
        except HfHubHTTPError as exc:
            raise OSError(f"Cannot fetch the LexVerse download manifest: {exc}. Use --manifest for a local YAML file.") from exc
        # The HF cache path is snapshots/<commit>/knowledge/sources.yaml.
        revision = path.parents[1].name
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Cannot determine the downloaded manifest's HF revision")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), revision
    except yaml.YAMLError as exc:
        raise ValueError("Invalid knowledge download YAML") from exc


def hosted_directory_files(url):
    from huggingface_hub import list_repo_files
    from huggingface_hub.errors import HfHubHTTPError

    repo_id, version_path = url.split("/datasets/", 1)[1].split("/resolve/", 1)
    revision, prefix = version_path.split("/", 1)
    prefix = unquote(prefix)
    try:
        files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    except HfHubHTTPError as exc:
        raise OSError(str(exc)) from exc
    matched = [name for name in files if name.startswith(prefix)]
    if not matched:
        raise ValueError(f"No hosted files found under {prefix}")
    return [(url + quote(name[len(prefix):], safe="/"), name[len(prefix):]) for name in matched]


def download_file(url, target, is_hf, *, cache_dir=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        if is_hf:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import HfHubHTTPError

            repo_id, version_path = url.split("/datasets/", 1)[1].split("/resolve/", 1)
            revision, filename = version_path.split("/", 1)
            try:
                cached = hf_hub_download(repo_id, unquote(filename), repo_type="dataset", revision=revision, cache_dir=cache_dir)
            except HfHubHTTPError as exc:
                raise OSError(str(exc)) from exc
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                with open(cached, "rb") as source:
                    shutil.copyfileobj(source, output)
        else:
            request = Request(url, headers={"User-Agent": "LexVerse-knowledge-downloader"})
            with urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    shutil.copyfileobj(response, output)
        with temporary.open("rb") as source:
            prefix = source.read(512)
        if not prefix or prefix.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("Received an empty file or a Git LFS pointer instead of data")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def add_arguments(parser):
    parser.add_argument("--manifest", type=Path, help="Local download YAML (default: fetch knowledge/sources.yaml from LexVerse HF)")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_ROOT, help="Root containing knowledge/ (default: <LexVerse root>/.lexverse/capabilities)")
    parser.add_argument("--sources", nargs="+", help="Download only the named knowledge sources")
    parser.add_argument("--dry-run", action="store_true", help="List data downloads; use --manifest to avoid fetching the HF manifest")


def run(args):
    cache_dir = str(args.local_dir.resolve() / "cache" / "huggingface")
    manifest, revision = load_manifest(args.manifest, cache_dir=cache_dir)
    plan = download_plan(manifest, args.local_dir, args.sources, revision)
    failed = False
    for name, url, target, is_hf in plan:
        if args.dry_run:
            suffix = "**" if url.endswith("/") else ""
            print(f"{name}: {url}{suffix}\n  -> {target}")
            continue
        try:
            files = [(url, target)]
            if url.endswith("/"):
                files = []
                for file_url, relative in hosted_directory_files(url):
                    file_target = (target / relative).resolve()
                    if not file_target.is_relative_to(target):
                        raise ValueError(f"Invalid hosted file path: {relative}")
                    files.append((file_url, file_target))
            for file_url, file_target in files:
                if file_target.exists():
                    print(f"SKIP {file_target} (already exists; not verified)")
                    continue
                print(f"DOWNLOAD {name}: {file_target}", flush=True)
                download_file(file_url, file_target, is_hf, cache_dir=cache_dir)
        except (OSError, ValueError, ImportError, HTTPError) as exc:
            failed = True
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            if is_hf:
                print("For gated datasets, accept access conditions on Hugging Face and run `hf auth login`.", file=sys.stderr)
    return 1 if failed else 0


def read_catalog(path: Path) -> dict:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        if catalog["schema_version"] != 1 or catalog["retrieval_config"] != RETRIEVAL:
            raise ValueError("Unsupported collection catalog")
        records = catalog["collections"]
        ids = set()
        for record in records:
            name, index_hash = record["id"], record["index_hash"]
            if (not isinstance(name, str) or not name or name in ids
                    or not isinstance(index_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", index_hash)
                    or record["path"] != "indexes/" + index_hash
                    or not isinstance(record["source_id"], str)
                    or type(record["chunk_count"]) is not int or record["chunk_count"] <= 0):
                raise ValueError("Invalid or duplicate collection entry")
            ids.add(name)
        return catalog
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"Cannot read knowledge collection catalog {path}: {exc}") from exc


def resolve_knowledge(sources: list[Path], base: Path, *, cache: Path | None = None) -> list[Path]:
    root = cache or DEFAULT_ROOT
    catalog = None
    resolved = []
    for source in sources:
        value = str(source)
        if value.startswith("collection:"):
            name = value.removeprefix("collection:")
            if catalog is None:
                if not (root / CATALOG).is_file():
                    raise ConfigError("No knowledge collection catalog; build local knowledge with "
                                      "lexverse environment build --knowledge <directory> or run "
                                      "lexverse environment download indexes")
                catalog = {record["id"]: record for record in read_catalog(root / CATALOG)["collections"]}
            if name not in catalog:
                raise ConfigError(f"Collection {name!r} is not downloaded or locally registered; "
                                  "build its source directory or run "
                                  f"lexverse environment download indexes --collections {name}")
            path = (root / catalog[name]["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not (path / "manifest.json").is_file():
                raise ConfigError(f"Collection {name!r} is missing; rebuild its source directory or run "
                                  f"lexverse environment download indexes --collections {name}")
        else:
            path = source.expanduser()
            path = (path if path.is_absolute() else base / path).resolve()
        if path not in resolved:
            resolved.append(path)
    return resolved


def _verify_collection(path, record):
    from lexverse.capabilities.knowledge.index import verify_index

    manifest = verify_index(path)
    required = {"documents.sqlite"}
    if manifest["identity"]["retrieval"]["mode"] != "keyword":
        required.update({"vectors.faiss", "vector_ids.json"})
    if (set(manifest["files"]) != required or manifest["index_hash"] != record["index_hash"]
            or manifest["identity"]["source_id"] != record["source_id"]
            or manifest["chunk_count"] != record["chunk_count"]):
        raise ConfigError(f"Index does not match collection {record['id']}")
    return manifest


def register_collection(source: Path, index: Path, config, *, cache: Path) -> dict:
    from lexverse.capabilities.knowledge.index import is_index_directory, verify_index
    from lexverse.capabilities.knowledge.loaders import KINDS
    from lexverse.runtime.results import atomic_write_json

    source, index, cache = source.resolve(), index.resolve(), cache.resolve()
    manifest = verify_index(index)
    name = manifest["source_name"]
    if is_index_directory(source):
        kinds = set(manifest["identity"]["kinds"].values())
        if len(kinds) == 1:
            kind = next(iter(kinds))
            if kind in KINDS and name != kind:
                name = kind + "/" + name
    else:
        for parent in (source, *source.parents):
            if parent.name in KINDS:
                name = source.relative_to(parent.parent).as_posix()
                break
    record = {"id": name, "source_id": manifest["identity"]["source_id"],
              "index_hash": manifest["index_hash"], "path": "indexes/" + manifest["index_hash"],
              "chunk_count": manifest["chunk_count"]}
    indexes = cache / "indexes"
    indexes.mkdir(parents=True, exist_ok=True)
    with FileLock(str(indexes / ".collections.lock")):
        installed = read_catalog(cache / CATALOG) if (cache / CATALOG).exists() else {
            "schema_version": 1, "retrieval_config": RETRIEVAL, "collections": [],
        }
        destination = cache / record["path"]
        if index != destination:
            with FileLock(str(destination) + ".lock"):
                if destination.exists():
                    _verify_collection(destination, record)
                else:
                    with tempfile.TemporaryDirectory(dir=indexes) as temporary:
                        staged = Path(temporary) / record["index_hash"]
                        staged.mkdir()
                        for filename in ("manifest.json", *manifest["files"]):
                            shutil.copyfile(index / filename, staged / filename)
                        _verify_collection(staged, record)
                        staged.rename(destination)
        entries = {entry["id"]: entry for entry in installed["collections"]}
        entries[name] = record
        (cache / RETRIEVAL).write_text(yaml.safe_dump(config.model_dump(mode="json"),
                                                   allow_unicode=True, sort_keys=False), encoding="utf-8")
        atomic_write_json(cache / CATALOG, {**installed, "collections": list(entries.values())})
    return record


def download_indexes(args):
    from huggingface_hub import hf_hub_download
    from lexverse.runtime.results import atomic_write_json

    root = args.local_dir.expanduser().resolve()
    cache = str(root / "cache/huggingface")
    catalog_path = Path(hf_hub_download(DATASET_REPO, CATALOG, repo_type="dataset",
                                      revision=args.revision, cache_dir=cache))
    # All files must come from the same immutable HF snapshot as the catalog.
    revision = catalog_path.parents[1].name
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConfigError("Cannot determine the collection catalog's HF revision")
    catalog = read_catalog(catalog_path)
    by_id = {record["id"]: record for record in catalog["collections"]}
    names = list(dict.fromkeys(args.collections)) if args.collections else list(by_id)
    unknown = set(names) - by_id.keys()
    if unknown:
        raise ConfigError("Unknown knowledge collections: " + ", ".join(sorted(unknown)))
    if args.dry_run:
        for name in names:
            print(f"{name}: {DATASET_REPO}@{revision}/{by_id[name]['path']} -> {root / by_id[name]['path']}")
        return 0
    indexes = root / "indexes"
    indexes.mkdir(parents=True, exist_ok=True)
    with FileLock(str(indexes / ".collections.lock")):
        installed = read_catalog(root / CATALOG) if (root / CATALOG).exists() else {
            "schema_version": 1, "retrieval_config": RETRIEVAL, "collections": [],
        }
        entries = {record["id"]: record for record in installed["collections"]}
        manifests = []
        for name in names:
            record = by_id[name]
            destination = root / record["path"]
            with FileLock(str(destination) + ".lock"):
                try:
                    manifest = _verify_collection(destination, record)
                except ConfigError:
                    with tempfile.TemporaryDirectory(dir=indexes) as temporary:
                        staged = Path(temporary) / record["index_hash"]
                        staged.mkdir()
                        remote = record["path"] + "/"
                        manifest_path = Path(hf_hub_download(DATASET_REPO, remote + "manifest.json",
                            repo_type="dataset", revision=revision, cache_dir=cache))
                        manifest = json.loads(manifest_path.read_text())
                        shutil.copyfile(manifest_path, staged / "manifest.json")
                        required = {"documents.sqlite"}
                        if manifest["identity"]["retrieval"]["mode"] != "keyword":
                            required.update({"vectors.faiss", "vector_ids.json"})
                        for filename in sorted(required):
                            downloaded = hf_hub_download(DATASET_REPO, remote + filename,
                                repo_type="dataset", revision=revision, cache_dir=cache)
                            shutil.copyfile(downloaded, staged / filename)
                        manifest = _verify_collection(staged, record)
                        if destination.exists():
                            shutil.rmtree(destination)
                        staged.rename(destination)
                    print(f"Downloaded and verified: {name} -> {destination}", flush=True)
                else:
                    print(f"Verified existing index: {name} -> {destination}", flush=True)
            entries[name] = record
            manifests.append(manifest)
        config_path = hf_hub_download(DATASET_REPO, RETRIEVAL, repo_type="dataset",
                                     revision=revision, cache_dir=cache)
        from lexverse.tasks.user import RetrievalConfig

        from lexverse.capabilities.knowledge.index import _build_recipe, _index_retrieval

        config = RetrievalConfig.model_validate(yaml.safe_load(Path(config_path).read_text()))
        for manifest in manifests:
            identity = manifest["identity"]
            if _build_recipe(identity["retrieval"]) != _index_retrieval(config, identity["embedding"]):
                raise ConfigError("Published retrieval configuration does not match the downloaded indexes")
        shutil.copyfile(config_path, root / RETRIEVAL)
        atomic_write_json(root / CATALOG, {**installed, "collections": list(entries.values())})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download knowledge from upstream and LexVerse HF")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, KeyError, ImportError, yaml.YAMLError) as exc:
        parser.exit(2, f"knowledge download error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
