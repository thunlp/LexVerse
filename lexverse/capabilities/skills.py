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

from filelock import FileLock
import yaml

from lexverse.capabilities.registry import ResourceRegistry, SkillSource
from lexverse.config import ConfigError
from lexverse.runtime.results import atomic_write_json


def tree_hashes(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ConfigError("Skill resources must not contain symbolic links")
        if path.is_file() and path.name != ".lexverse-source.json":
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def source_root(name: str, source: SkillSource, cache: Path) -> Path:
    if source.local_path is not None:
        if not source.local_path.is_dir():
            raise ConfigError(f"Local skill source does not exist: {name}")
        tree_hashes(source.local_path)
        return source.local_path
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
        try:
            manifest = json.loads((root / ".lexverse-source.json").read_text())
            valid = (manifest["commit"] == source.commit and manifest["repository"] == source.repository
                     and manifest["files"] == tree_hashes(root))
        except (OSError, ValueError, KeyError) as exc:
            raise ConfigError(f"Invalid skill cache: {name}") from exc
        if not valid:
            raise ConfigError(f"Skill cache identity mismatch: {name}")
    return root


def prepare_skills(registry: ResourceRegistry, selected: list[str], cache: Path,
                   agent_root: Path, *, interactive: bool) -> dict:
    destination = agent_root / "skills"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    names = set()
    folders = set()
    roots = {}
    needs_confirmation = False
    for skill_id in selected:
        definition = registry.skills.get(skill_id)
        if definition is None:
            raise ConfigError(f"Unknown skill: {skill_id}")
        if definition.requires_confirmation and not interactive:
            raise ConfigError(f"Skill {skill_id} requires interactive fact confirmation")
        needs_confirmation |= definition.requires_confirmation
        if definition.source not in roots:
            roots[definition.source] = source_root(definition.source, registry.skill_sources[definition.source], cache)
        root = roots[definition.source]
        folder = root / definition.path
        skill_file = folder / "SKILL.md"
        if not skill_file.is_file():
            raise ConfigError(f"Missing SKILL.md: {skill_id}")
        content = skill_file.read_text(encoding="utf-8")
        match = re.match(r"\A---\s*\n(.*?)\n---(?:\n|$)", content, re.S)
        metadata = yaml.safe_load(match.group(1)) if match else None
        if not isinstance(metadata, dict) or not all(isinstance(metadata.get(k), str) and metadata[k].strip() for k in ("name", "description")):
            raise ConfigError(f"Invalid skill frontmatter: {skill_id}")
        if metadata["name"] in names:
            raise ConfigError(f"Duplicate skill name: {metadata['name']}")
        names.add(metadata["name"])
        target = destination / definition.source / folder.relative_to(root)
        if target in folders:
            raise ConfigError(f"Duplicate skill folder: {folder.name}")
        folders.add(target)
        shutil.copytree(folder, target, dirs_exist_ok=True)
        _copy_references(root, folder, target, agent_root / "skills")
        hashes = tree_hashes(target)
        records.append({
            "id": skill_id, "name": metadata["name"], "description": metadata["description"],
            "path": "/" + (target / "SKILL.md").relative_to(agent_root).as_posix(), "source": definition.source,
            "commit": registry.skill_sources[definition.source].commit, "files": hashes,
            "requires_confirmation": definition.requires_confirmation,
        })
    exposed = {p.parent for p in destination.rglob("SKILL.md")}
    if exposed != folders:
        raise ConfigError("Skill references expose unselected skill instructions")
    return {"skills": records, "requires_confirmation": needs_confirmation,
            "paths": ["/" + folder.relative_to(agent_root).as_posix() + "/" for folder in sorted({target.parent for target in folders})], "files": tree_hashes(agent_root / "skills")}


def skill_bindings(skills, indexes, remote_tools, selected_tools) -> dict:
    sources = [{"id": index.source_id, "name": index.manifest["source_name"],
                "kinds": sorted(row[0] for row in index.db.execute(
                    "SELECT DISTINCT json_extract(metadata, '$.kind') FROM chunks"))}
               for index in indexes]
    operations = {}
    for operation, kind, services in (
        ("law_search", "legal_laws", {"pkulaw_law_search", "pkulaw_law_keyword", "pkulaw_law_item_keyword", "pkulaw_law_agg"}),
        ("case_search", "legal_cases", {"pkulaw_case_search", "pkulaw_case_keyword"}),
        ("norm_validity", "legal_laws", {"pkulaw_statute_history", "pkulaw_law_recognition", "pkulaw_citation_validator"}),
    ):
        local = [source["id"] for source in sources if kind in source["kinds"]]
        tools = [tool.name for tool in remote_tools if (tool.metadata or {}).get("service") in services]
        if local:
            tools += [name for name in ("search_knowledge", "fetch_knowledge") if name in selected_tools]
        operations[operation] = {"tools": tools, "knowledge_sources": local}
    return {"sources": sources, "operations": operations,
            "generic_knowledge_tools": [name for name in ("search_knowledge", "fetch_knowledge")
                                        if indexes and name in selected_tools],
            "skills": [{"id": record["id"], "requires_fact_confirmation": record["requires_confirmation"]}
                       for record in skills["skills"]],
            "fallbacks": {key: value for key, value in {
                "case-retrieval": "Without case sources, design queries and mark concrete cases pending retrieval.",
                "legal-norm-validity-check": "Without verifiable legal sources, mark validity uncertain and pending manual verification.",
            }.items() if key in {record["id"] for record in skills["skills"]}},
            "limitations": "Generic user knowledge is not automatically an authoritative law or case source. Local law snapshots do not prove current validity. Tools are selected by need."}


def _copy_references(source_root: Path, folder: Path, target: Path, projection_root: Path) -> None:
    pending = [(p, target / p.relative_to(folder)) for p in folder.rglob("*.md")]
    seen = set()
    while pending:
        source, projected = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        for link in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            link = link.split("#", 1)[0]
            if not link or ":" in link or link.startswith("/"):
                continue
            original = (source.parent / link).resolve()
            output = (projected.parent / link).resolve()
            if not original.is_relative_to(source_root.resolve()) or not output.is_relative_to(projection_root.resolve()):
                raise ConfigError("Skill reference escapes its resource projection")
            if not original.is_file():
                raise ConfigError(f"Missing skill reference: {link}")
            if original.name == "SKILL.md" and not output.exists():
                raise ConfigError("Skill reference exposes unselected skill instructions")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                shutil.copyfile(original, output)
            if original.suffix == ".md":
                pending.append((original, output))
