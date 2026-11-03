# Copyright 2025-2026 Alex Shaw and the Harbor Framework Team
# Portions of this file are derived from Harbor (https://github.com/laude-institute/harbor, v0.20.0).
# Modified by the LexVerse Contributors for LexVerse package integration and legal capability support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TaskOS(str, Enum):
    LINUX = "linux"
    WINDOWS = "windows"


class EnvironmentType(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["sse", "streamable-http", "stdio"] = "streamable-http"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required when transport is '{self.transport}'.")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required when transport is 'stdio'.")
        return self


class TaskEnvConfig(BaseModel):
    os: TaskOS = TaskOS.LINUX
    cpus: int = Field(default=1, gt=0, strict=True)
    memory_mb: int = Field(default=2048, gt=0, strict=True)
    storage_mb: int = Field(default=10240, gt=0, strict=True)
    gpus: int = Field(default=0, ge=0, strict=True)
    gpu_types: list[str] | None = None
    tpus: int = Field(default=0, ge=0, strict=True)
    tpu_type: str | None = None
    allow_internet: bool = True
    network_allowlist: list[str] | None = None
    network_mode: Literal["public", "allowlist"] = "public"
    allowed_hosts: list[str] = Field(default_factory=list)
    docker_image: str | None = None
    workdir: str | None = None
    skills_dir: Literal["/skills"] | None = "/skills"
    memory_dir: str | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    build_timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
