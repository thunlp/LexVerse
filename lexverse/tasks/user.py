from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lexverse.config import ConfigError, ModelConfig, parse_model_config


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class AgentLimits(StrictModel):
    model_calls: int = Field(default=30, gt=0, strict=True)
    tool_calls: int = Field(default=60, gt=0, strict=True)
    active_timeout_sec: float = Field(default=900, gt=0, allow_inf_nan=False)
    tool_timeout_sec: float = Field(default=60, gt=0, allow_inf_nan=False)


class TaskCapabilities(StrictModel):
    registry: Path | None = None
    knowledge: list[Path] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mcp: list[str] = Field(default_factory=list)
    tools: list[Literal["search_knowledge", "fetch_knowledge", "ask_user"]] = Field(
        default_factory=lambda: ["search_knowledge", "fetch_knowledge", "ask_user"]
    )


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class EmbeddingConfig(StrictModel):
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL, min_length=1)
    revision: str | None = None
    device: str = Field(default="cpu", min_length=1)
    batch_size: int = Field(default=32, gt=0, strict=True)
    query_prefix: str | None = None
    document_prefix: str = ""
    max_tokens: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def defaults(self):
        if not self.model.strip() or not self.device.strip():
            raise ValueError("Embedding model and device must contain text")
        if self.model == DEFAULT_EMBEDDING_MODEL and self.revision is None:
            self.revision = DEFAULT_EMBEDDING_REVISION
        if self.query_prefix is None:
            self.query_prefix = DEFAULT_QUERY_PREFIX if self.model == DEFAULT_EMBEDDING_MODEL else ""
        return self

    def resolve_path(self, base: Path):
        candidate = Path(self.model).expanduser()
        if self.model.startswith(("/", ".", "~")) or (base / candidate).is_dir():
            self.model = str(_resolve(candidate, base))
            if not Path(self.model).is_dir():
                raise ConfigError(f"Embedding model directory does not exist: {self.model}")

    def uses_default_encoding(self):
        return (self.model == DEFAULT_EMBEDDING_MODEL and self.revision == DEFAULT_EMBEDDING_REVISION
                and self.query_prefix == DEFAULT_QUERY_PREFIX and not self.document_prefix
                and self.max_tokens in (None, 512))


class RetrievalConfig(StrictModel):
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunk_tokens: int = Field(default=400, gt=0, strict=True)
    overlap_tokens: int = Field(default=60, ge=0, strict=True)
    keyword_candidates: int = Field(default=40, gt=0, strict=True)
    vector_candidates: int = Field(default=40, gt=0, strict=True)
    top_k: int = Field(default=8, ge=1, le=20, strict=True)


class OutputContract(StrictModel):
    required: list[str] = Field(default_factory=list)
    allowed_formats: list[Literal["md", "json", "csv", "txt"]] = Field(
        default_factory=lambda: ["md", "json", "csv", "txt"]
    )

    @field_validator("required")
    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        for value in paths:
            path = PurePosixPath(value)
            if ("\\" in value or path.is_absolute() or ".." in path.parts
                    or len(path.parts) < 2 or path.parts[0] != "artifacts"):
                raise ValueError("required outputs must be under artifacts/")
        return paths


class KnowledgeFilters(StrictModel):
    kind: Literal["legal_laws", "legal_cases", "legal_concepts", "legal_qa", "legal_templates", "user"] | None = None
    title: str | None = Field(default=None, min_length=1, max_length=300)
    article: str | None = Field(default=None, min_length=1, max_length=100)
    page: int | None = Field(default=None, gt=0, strict=True)


class UserTaskSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    instruction: str = Field(min_length=1)
    inputs: Path
    model: dict
    context_window_tokens: int | None = Field(default=None, gt=0, strict=True)
    capabilities: TaskCapabilities = Field(default_factory=TaskCapabilities)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    limits: AgentLimits = Field(default_factory=AgentLimits)
    interactive: bool = True
    outputs: OutputContract = Field(default_factory=OutputContract)

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must contain text")
        return value

    def model_config_value(self, base_dir: Path) -> ModelConfig:
        return parse_model_config(self.model, field="model", base_dir=base_dir)


def load_user_task(path: Path, registry: Path | None = None) -> UserTaskSpec:
    path = path.expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        _reject_credentials(data)
        task = UserTaskSpec.model_validate(data)
        task.model_config_value(path.parent)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid user task: {exc}") from exc
    if task.retrieval.overlap_tokens >= task.retrieval.chunk_tokens:
        raise ConfigError("retrieval.overlap_tokens must be smaller than chunk_tokens")
    task.inputs = _resolve(task.inputs, path.parent)
    task.retrieval.embedding.resolve_path(path.parent)
    task.capabilities.knowledge = [_resolve(p, path.parent) for p in task.capabilities.knowledge]
    if registry is not None:
        task.capabilities.registry = registry.expanduser().resolve()
    elif task.capabilities.registry is not None:
        task.capabilities.registry = _resolve(task.capabilities.registry, path.parent)
    if not task.inputs.is_dir():
        raise ConfigError(f"Input directory does not exist: {task.inputs}")
    for source in task.capabilities.knowledge:
        if not source.is_dir():
            raise ConfigError(f"Knowledge directory does not exist: {source}")
    for name in task.outputs.required:
        if PurePosixPath(name).suffix.lstrip(".") not in task.outputs.allowed_formats:
            raise ConfigError(f"Output format is not allowed: {name}")
    return task


def _resolve(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _reject_credentials(value):
    if isinstance(value, dict):
        if any(str(key).lower() in {"api_key", "openai_api_key", "token", "hf_token", "authorization", "password", "headers"} for key in value):
            raise ConfigError("Credentials must be stored in host secrets, not task configuration")
        for item in value.values():
            _reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _reject_credentials(item)
