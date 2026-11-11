from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lexverse.config import ConfigError
from lexverse.models.task.environment import TaskEnvConfig
from lexverse.capabilities.resources import MCPService


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class AgentLimits(StrictModel):
    model_calls: int = Field(default=30, gt=0, strict=True)
    tool_calls: int = Field(default=60, gt=0, strict=True)
    active_timeout_sec: float = Field(default=900.0, gt=0, allow_inf_nan=False)
    tool_timeout_sec: float = Field(default=60.0, gt=0, allow_inf_nan=False)


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_QUERY_PREFIX = ""


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


class RetrievalConfig(StrictModel):
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunk_tokens: int = Field(default=400, gt=0, strict=True)
    chunk_tokens_by_kind: dict[
        Literal["legal_laws", "legal_cases", "legal_concepts", "legal_qa", "legal_templates", "user"],
        Annotated[int, Field(gt=0, strict=True)],
    ] = Field(default_factory=dict)
    overlap_tokens: int = Field(default=60, ge=0, strict=True)
    keyword_candidates: int = Field(default=100, gt=0, strict=True)
    vector_candidates: int = Field(default=100, gt=0, strict=True)
    top_k: int = Field(default=8, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def chunk_limits(self):
        if self.overlap_tokens >= min([self.chunk_tokens, *self.chunk_tokens_by_kind.values()]):
            raise ValueError("retrieval overlap must be smaller than every chunk size")
        return self


class KnowledgeFilters(StrictModel):
    kind: Literal["legal_laws", "legal_cases", "legal_concepts", "legal_qa", "legal_templates", "user"] | None = None
    title: str | None = Field(default=None, min_length=1, max_length=300)
    article: str | None = Field(default=None, min_length=1, max_length=100)
    page: int | None = Field(default=None, gt=0, strict=True)


class ArtifactSpec(StrictModel):
    source: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    exclude: list[str] = Field(default_factory=list)
    service: str = "main"
    required: bool = True

    @field_validator("source")
    @classmethod
    def source_path(cls, value):
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or "\\" in value or value == "/":
            raise ValueError("Artifact source must be an absolute environment path, not its root")
        return value

    @field_validator("destination")
    @classmethod
    def destination_path(cls, value):
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or value in {".", ""}:
            raise ValueError("Artifact destination must stay inside the Trial artifacts directory")
        return value


class TaskEnvironment(TaskEnvConfig):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    skills: list[str] = Field(default_factory=list)
    mcp_servers: list[MCPService] = Field(default_factory=list)
    knowledge: list[Path] = Field(default_factory=list)

    @field_validator("skills")
    @classmethod
    def environment_skill_paths(cls, values):
        if any(not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts for value in values):
            raise ValueError("Task Skills must use absolute environment paths")
        return values


class VerifierSpec(StrictModel):
    environment_mode: Literal["shared", "separate"] | None = None
    environment: TaskEnvironment | None = None
    timeout_sec: float = Field(default=300.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def resolve_mode(self):
        if self.environment_mode == "shared" and self.environment is not None:
            raise ValueError("shared verifier cannot declare a separate environment")
        if self.environment_mode is None:
            self.environment_mode = "separate" if self.environment is not None else "shared"
        return self


class TaskAgent(StrictModel):
    timeout_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class UserTaskSpec(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    agent: TaskAgent = Field(default_factory=TaskAgent)
    environment: TaskEnvironment = Field(default_factory=TaskEnvironment)
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    verifier: VerifierSpec | None = None

    @model_validator(mode="after")
    def unique_destinations(self):
        destinations = [PurePosixPath(item.destination) for item in self.artifacts]
        for index, path in enumerate(destinations):
            if any(path == other or path.is_relative_to(other) or other.is_relative_to(path)
                   for other in destinations[:index]):
                raise ValueError("Artifact destinations must not overlap")
        return self


def _resolve(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return (path if path.is_absolute() else base / path).resolve()
