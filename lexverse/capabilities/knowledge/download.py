import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

import yaml

from lexverse.runtime.paths import CAPABILITIES_ROOT as DEFAULT_ROOT

DATASET_REPO = "TsinghuaNLP/LexVerse"


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
