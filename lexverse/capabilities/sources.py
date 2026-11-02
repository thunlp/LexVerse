"""Source manifests used only by the Skill downloader, not execution aliases."""
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lexverse.config import ConfigError


class SkillSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str | None = None
    commit: str | None = None
    local_path: Path | None = None
    path: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value):
        if value is not None and (not value or Path(value).is_absolute() or ".." in Path(value).parts or "\\" in value or value == "."):
            raise ValueError("Skill source path must be a relative directory")
        return value


class SkillSources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_sources: dict[str, SkillSource] = Field(default_factory=dict)


def validate_sources(data: dict) -> SkillSources:
    sources = SkillSources.model_validate(data)
    for name, source in sources.skill_sources.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ConfigError("Skill source IDs must be safe path components")
        if source.local_path is None and not (
            source.repository and re.fullmatch(r"https://github.com/[\w.-]+/[\w.-]+", source.repository)
            and source.commit and re.fullmatch(r"[0-9a-f]{40}", source.commit)
        ):
            raise ConfigError(f"Skill source {name} requires a local path or GitHub repository and full commit")
    return sources
