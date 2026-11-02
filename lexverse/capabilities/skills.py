from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from urllib.request import urlopen
from urllib.parse import quote

from filelock import FileLock
import yaml

from lexverse.capabilities.sources import SkillSources, SkillSource, validate_sources
from lexverse.config import ConfigError
from lexverse.utils.files import atomic_write_json


def tree_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ConfigError("Skill resources must not contain symbolic links")
        if path.is_file() and path.name != ".lexverse-source.json":
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _verify_source_cache(root: Path, source: SkillSource, name: str) -> None:
    try:
        manifest = json.loads((root / ".lexverse-source.json").read_text())
        valid = (manifest["repository"] == source.repository and manifest["commit"] == source.commit
                 and (source.path is None or manifest["path"] == source.path)
                 and manifest["files"] == tree_hashes(root))
    except (OSError, ValueError, KeyError) as exc:
        raise ConfigError(f"Invalid skill cache: {name}") from exc
    if not valid:
        raise ConfigError(f"Skill cache identity mismatch: {name}")


def source_root(name: str, source: SkillSource, cache: Path) -> Path:
    if source.local_path is not None:
        if not source.local_path.is_dir():
            raise ConfigError(f"Local skill source does not exist: {name}")
        tree_hashes(source.local_path)
        return source.local_path
    if source.path is not None:
        return _directory_root(name, source, cache)
    root = cache / "skills" / name / source.commit
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock"):
        if not root.exists():
            repository = source.repository.removeprefix("https://github.com/")
            url = f"https://codeload.github.com/{repository}/tar.gz/{source.commit}"
            with urlopen(url, timeout=60) as response:
                archive = response.read()
            with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
                staged = Path(temporary) / "source"
                staged.mkdir()
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
                    for member in bundle.getmembers():
                        parts = Path(member.name).parts
                        relative = Path(*parts[1:])
                        if member.issym() or member.islnk() or ".." in parts or Path(member.name).is_absolute():
                            raise ConfigError("Unsafe skill archive entry")
                        if member.isdir():
                            (staged / relative).mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            target = staged / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.extractfile(member) as content:
                                target.write_bytes(content.read())
                        else:
                            raise ConfigError("Unsupported skill archive entry")
                atomic_write_json(staged / ".lexverse-source.json", {
                    "repository": source.repository, "commit": source.commit,
                    "archive_hash": hashlib.sha256(archive).hexdigest(), "files": tree_hashes(staged),
                })
                staged.rename(root)
        _verify_source_cache(root, source, name)
    return root


def _directory_root(name: str, source: SkillSource, cache: Path) -> Path:
    identity = hashlib.sha256(source.path.encode()).hexdigest()[:12]
    root = cache / "skills" / name / f"{source.commit}-{identity}"
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock"):
        if not root.exists():
            repository = source.repository.removeprefix("https://github.com/")
            tree = source.commit
            for part in Path(source.path).parts:
                with urlopen(f"https://api.github.com/repos/{repository}/git/trees/{tree}", timeout=60) as response:
                    listing = json.load(response)
                directory = next((entry for entry in listing["tree"]
                                  if entry["path"] == part and entry["type"] == "tree"), None)
                if directory is None:
                    raise ConfigError(f"Missing skill source directory: {source.path}")
                tree = directory["sha"]
            with urlopen(f"https://api.github.com/repos/{repository}/git/trees/{tree}?recursive=1", timeout=60) as response:
                listing = json.load(response)
            if listing.get("truncated"):
                raise ConfigError(f"Incomplete GitHub file listing: {name}")
            prefix = source.path.rstrip("/") + "/"
            entries = [entry for entry in listing["tree"] if entry["type"] != "tree"]
            if not entries:
                raise ConfigError(f"Empty or missing skill source directory: {source.path}")
            with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
                staged = Path(temporary) / "source"
                staged.mkdir()
                for entry in entries:
                    relative = Path(entry["path"])
                    if (relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]
                            or entry["type"] != "blob" or entry["mode"] not in {"100644", "100755"}):
                        raise ConfigError("Unsafe skill directory entry")
                    target = staged / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    url = f"https://raw.githubusercontent.com/{repository}/{source.commit}/{quote(prefix + entry['path'], safe='/')}"
                    with urlopen(url, timeout=60) as response:
                        target.write_bytes(response.read())
                    if entry["mode"] == "100755":
                        target.chmod(0o755)
                atomic_write_json(staged / ".lexverse-source.json", {
                    "repository": source.repository, "commit": source.commit, "path": source.path,
                    "files": tree_hashes(staged),
                })
                staged.rename(root)
        _verify_source_cache(root, source, name)
    return root


def load_skill_sources(path: Path | None = None, *, cache_dir=None) -> SkillSources:
    if path is None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError

        try:
            path = Path(hf_hub_download("TsinghuaNLP/LexVerse", "skills/sources.yaml", repo_type="dataset", cache_dir=cache_dir))
        except HfHubHTTPError as exc:
            raise OSError(f"Cannot fetch the LexVerse skills manifest: {exc}. Use --manifest for a local YAML file.") from exc
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("Invalid skills download YAML") from exc
    if not isinstance(manifest, dict):
        raise ConfigError("Skills download manifest must be a source-name mapping")
    sources = {}
    for name, entry in manifest.items():
        if not isinstance(entry, dict) or set(entry) - {"delivery", "upstream_url", "upstream_revision", "upstream_path"}:
            raise ConfigError(f"Invalid skills download source: {name}")
        if entry.get("delivery", "upstream") != "upstream":
            raise ConfigError(f"Unsupported skills delivery type: {name}")
        sources[name] = {"repository": entry.get("upstream_url"), "commit": entry.get("upstream_revision"),
                         "path": entry.get("upstream_path")}
    return validate_sources({"skill_sources": sources})


def download_skills(registry: SkillSources, selected: list[str] | None, cache: Path, *, dry_run=False) -> int:
    selected = list(dict.fromkeys(selected)) if selected else list(registry.skill_sources)
    unknown = set(selected) - registry.skill_sources.keys()
    if unknown:
        raise ConfigError("Unknown skill sources: " + ", ".join(sorted(unknown)))
    for name in selected:
        source = registry.skill_sources[name]
        if dry_run:
            location = source.local_path or f"{source.repository}@{source.commit}"
            if source.path is not None:
                location = f"{location}/{source.path}"
            identity = "-" + hashlib.sha256(source.path.encode()).hexdigest()[:12] if source.path else ""
            target = source.local_path or cache / "skills" / name / (source.commit + identity)
            print(f"{name}: {location}\n  -> {target}")
            continue
        root = source_root(name, source, cache)
        skill_files = sorted(path for path in root.rglob("SKILL.md") if path.is_file())
        print(f"Skill source ready: {name}; skills={len(skill_files)}; path={root}")
        for skill_file in skill_files:
            print(f"  Skill found: {skill_file.parent.relative_to(root)}")
    return 0


def prepare_skills(selected, cache: Path, agent_root: Path) -> dict:
    """Project explicit Skill directories; a later source replaces the same directory name."""
    destination = agent_root / "skills"
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    overrides = []
    for reference in selected:
        if reference.repository:
            source = SkillSource(repository=reference.repository, commit=reference.commit,
                                 path=reference.path.as_posix())
            source_id = hashlib.sha256(reference.repository.encode()).hexdigest()[:16]
            folder = source_root(source_id, source, cache)
            name = reference.path.name
        else:
            folder = reference.path
            name = folder.name
        if not name or name in {".", ".."}:
            raise ConfigError("Skill directory needs a name")
        skill_file = folder / "SKILL.md"
        if not skill_file.is_file():
            raise ConfigError(f"Missing SKILL.md: {folder}")
        tree_hashes(folder)
        content = skill_file.read_text(encoding="utf-8")
        match = re.match(r"\A---\s*\n(.*?)\n---(?:\n|$)", content, re.S)
        metadata = yaml.safe_load(match.group(1)) if match else None
        if not isinstance(metadata, dict) or not all(isinstance(metadata.get(k), str) and metadata[k].strip()
                                                    for k in ("name", "description")):
            raise ConfigError(f"Invalid skill frontmatter: {folder}")
        target = destination / name
        if name in records:
            overrides.append({"name": name, "previous": records[name]["source"],
                              "replacement": reference.model_dump(mode="json")})
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(folder, target)
        records[name] = {"id": name, "name": name, "description": metadata["description"],
                         "path": f"/skills/{name}/SKILL.md", "source": reference.model_dump(mode="json"),
                         "commit": reference.commit, "files": tree_hashes(target)}
    return {"skills": list(records.values()), "paths": ["/skills/"],
            "overrides": overrides, "files": tree_hashes(destination)}


async def prepare_task_skills(environment, spec, resources, cache: Path):
    if not spec.task_skills:
        return
    from lexverse.capabilities.resources import SkillRef
    from lexverse.capabilities.base import identity

    with tempfile.TemporaryDirectory(prefix="task-skills-") as temporary:
        fixed, origins = [], {}
        for index, source in enumerate(spec.task_skills):
            root = Path(temporary) / str(index) / Path(source).name
            await environment.download_task_skills(source, root)
            tree_hashes(root)
            folders = [root] if (root / "SKILL.md").is_file() else sorted(path.parent for path in root.rglob("SKILL.md"))
            if not folders:
                raise ConfigError(f"Task Skill directory contains no SKILL.md: {source}")
            for folder in folders:
                fixed.append(SkillRef(path=folder))
                origins[str(folder)] = {"environment_path": str(Path(source) / folder.relative_to(root))}
        prepared = prepare_skills([*fixed, *spec.selection.skills], cache, spec.run_dir / ".internal/agent_fs")

        def provenance(value):
            if isinstance(value, dict):
                if value.get("path") in origins:
                    return origins[value["path"]]
                return {key: provenance(item) for key, item in value.items()}
            if isinstance(value, list):
                return [provenance(item) for item in value]
            return value

        prepared = provenance(prepared)
        prepared["execution_files"] = resources["skills"].get("execution_files", resources["skills"]["files"])
        resources["skills"] = prepared
        manifest_path = spec.run_dir / ".internal/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(resources=resources, resource_hash=identity(resources))
        atomic_write_json(manifest_path, manifest)
