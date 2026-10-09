from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import parse_qsl, urlparse

from pydantic import Field, field_validator
import yaml

from lexverse.config import ConfigError
from lexverse.tasks.user import StrictModel


class SkillSource(StrictModel):
    repository: str | None = None
    commit: str | None = None
    local_path: Path | None = None


class SkillDefinition(StrictModel):
    source: str
    path: str
    requires_confirmation: bool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("Skill path must be relative to its source")
        return value


class MCPService(StrictModel):
    url: str
    credential_ref: str | None = None
    required: bool = True
    allowed_tools: list[str] | None = None
    timeout_sec: float = Field(default=60, gt=0, allow_inf_nan=False)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value):
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("MCP URL must be an HTTP endpoint without credentials")
        credential_fields = {"apikey", "token", "accesstoken", "authorization", "password", "secret"}
        if parsed.fragment or any(key.lower().replace("_", "").replace("-", "") in credential_fields
                                  for key, _ in parse_qsl(parsed.query)):
            raise ValueError("MCP URL must not contain credentials or fragments; use credential_ref")
        return value


class ResourceRegistry(StrictModel):
    schema_version: int = 1
    skill_sources: dict[str, SkillSource] = Field(default_factory=dict)
    skills: dict[str, SkillDefinition] = Field(default_factory=dict)
    mcp: dict[str, MCPService] = Field(default_factory=dict)


def load_registry(path: Path | None = None) -> ResourceRegistry:
    data = yaml.safe_load(Path(__file__).with_name("catalog.yaml").read_text(encoding="utf-8"))
    if path is not None:
        try:
            override = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"Cannot read capability registry: {path}") from exc
        if not isinstance(override, dict) or set(override) - set(data):
            raise ConfigError("Registry must use schema_version, skill_sources, skills and mcp")
        for section, entries in override.items():
            if section == "schema_version":
                data[section] = entries
                continue
            if not isinstance(entries, dict):
                raise ConfigError(f"registry.{section} must be a mapping")
            for name, fields in entries.items():
                if not isinstance(fields, dict):
                    raise ConfigError(f"registry.{section}.{name} must be a mapping")
                merged = {**data[section].get(name, {}), **fields}
                if section == "skill_sources" and merged.get("local_path"):
                    local = Path(merged["local_path"]).expanduser()
                    merged["local_path"] = str((path.parent / local).resolve() if not local.is_absolute() else local.resolve())
                    merged["repository"] = None
                    merged["commit"] = None
                data[section][name] = merged
    try:
        registry = ResourceRegistry.model_validate(data)
    except ValueError as exc:
        raise ConfigError(f"Invalid capability registry: {exc}") from exc
    if registry.schema_version != 1:
        raise ConfigError("Unsupported registry schema version")
    for name, source in registry.skill_sources.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ConfigError("Skill source IDs must be safe path components")
        if source.local_path is None and not (
            source.repository and re.fullmatch(r"https://github.com/[\w.-]+/[\w.-]+", source.repository)
            and source.commit and re.fullmatch(r"[0-9a-f]{40}", source.commit)
        ):
            raise ConfigError(f"Skill source {name} requires a local path or GitHub repository and full commit")
    for name, definition in registry.skills.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or definition.source not in registry.skill_sources:
            raise ConfigError(f"Invalid skill definition: {name}")
    for name in registry.mcp:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ConfigError(f"Invalid MCP alias: {name}")
    return registry
