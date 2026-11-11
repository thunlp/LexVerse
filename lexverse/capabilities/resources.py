from pathlib import Path
import re
from typing import Literal
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SkillRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    repository: str | None = None
    commit: str | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.repository:
            if not re.fullmatch(r"https://github.com/[\w.-]+/[\w.-]+", self.repository):
                raise ValueError("Skill repository must be a GitHub HTTPS URL")
            if not self.commit or not re.fullmatch(r"[0-9a-f]{40}", self.commit):
                raise ValueError("Git Skills require a full commit")
            if self.path.is_absolute() or ".." in self.path.parts or str(self.path) == ".":
                raise ValueError("Git Skill path must be a repository subdirectory")
        elif self.commit is not None:
            raise ValueError("A local Skill does not accept commit")
        return self


class MCPService(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    transport: Literal["sse", "streamable-http", "stdio"] = "streamable-http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    credential_ref: str | None = None
    required: bool = True
    allowed_tools: list[str] | None = None
    timeout_sec: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_connection(self):
        if self.transport == "stdio":
            if not self.command or self.url or self.credential_ref:
                raise ValueError("stdio needs command; URL and HTTP credential_ref are unsupported")
        else:
            parsed = urlparse(self.url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("MCP URL must be an HTTP endpoint without embedded credentials")
            sensitive = {"apikey", "token", "accesstoken", "authorization", "password", "secret"}
            if parsed.fragment or any(key.lower().replace("_", "").replace("-", "") in sensitive
                                      for key, _ in parse_qsl(parsed.query)):
                raise ValueError("Use credential_ref instead of credentials in MCP URLs")
            if self.command or self.args or self.env:
                raise ValueError("HTTP MCP does not accept process configuration")
        return self


class ResourceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    knowledge: list[Path] = Field(default_factory=list)
    skills: list[SkillRef] = Field(default_factory=list)
    mcp_servers: list[MCPService] = Field(default_factory=list)
    # Vanilla built-in tools to remove. Configured skills require read_file.
    disabled_tools: list[Literal["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
                                 "write_todos"]] = Field(default_factory=list)
    tools: list[Literal["search_knowledge", "fetch_knowledge", "ask_user"]] = Field(
        default_factory=lambda: ["search_knowledge", "fetch_knowledge", "ask_user"])

    @field_validator("skills", mode="before")
    @classmethod
    def local_paths(cls, values):
        if not isinstance(values, list):
            return values
        return [{"path": value} if isinstance(value, (str, Path)) else value for value in values]

    def resolve(self, base: Path):
        from lexverse.capabilities.knowledge.download import resolve_knowledge
        self.knowledge = resolve_knowledge(self.knowledge, base)
        for skill in self.skills:
            if skill.repository is None:
                path = skill.path.expanduser()
                skill.path = (path if path.is_absolute() else base / path).resolve()
        return self


def merge_mcp(task: list[MCPService], execution: list[MCPService]) -> list[MCPService]:
    """Reject ambiguous names across task-owned and experiment-owned services."""
    from lexverse.config import ConfigError
    names = [service.name for service in [*task, *execution]]
    if len(names) != len(set(names)):
        raise ConfigError("Task and Job MCP services must have distinct names")
    return [*task, *execution]
