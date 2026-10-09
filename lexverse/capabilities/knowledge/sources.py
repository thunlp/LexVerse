from pathlib import Path
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

from lexverse.config import ConfigError


MANIFEST_NAME = ".lexverse-knowledge.yaml"


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    kind: Literal["user", "legal_laws", "legal_cases", "legal_concepts", "legal_qa", "legal_templates"] = "user"
    text_fields: list[str] = Field(min_length=1)
    title_field: str | None = None
    metadata_fields: dict[Literal["article", "effective_from", "effective_to", "source"], str] = Field(default_factory=dict)
    exclude_fields: list[str] = Field(default_factory=list)


def read_source_manifest(root: Path, expected_hash: str | None = None) -> SourceManifest | None:
    path = root / MANIFEST_NAME
    if not path.exists():
        if expected_hash is not None:
            raise ConfigError("Knowledge manifest changed after scanning")
        return None
    if path.is_symlink():
        raise ConfigError("Knowledge manifest must not be a symbolic link")
    try:
        content = path.read_bytes()
        if expected_hash is not None and hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Knowledge manifest changed after scanning")
        manifest = SourceManifest.model_validate(yaml.safe_load(content.decode("utf-8")))
        fields = [*manifest.text_fields, *manifest.metadata_fields.values()]
        if manifest.title_field is not None:
            fields.append(manifest.title_field)
        if any(not field.strip() or any(not part for part in field.split(".")) for field in fields):
            raise ValueError("Field paths must contain nonempty names")
        if any(not field.strip() or "." in field for field in manifest.exclude_fields):
            raise ValueError("exclude_fields accepts recursive field names, not paths")
        return manifest
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid knowledge source manifest: {exc}") from exc


def field_value(record, path: str):
    value = record
    for name in path.split("."):
        if not isinstance(value, dict) or name not in value:
            raise ValueError(f"Mapped field is missing: {path}")
        value = value[name]
    return value
