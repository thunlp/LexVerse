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

from typing import Any
from dataclasses import asdict, dataclass, field
from pydantic import BaseModel, Field
from lexverse.utils.identity import identity


class AgentConfig(BaseModel):
    name: str | None = "vanilla"
    import_path: str | None = None
    model_name: str | None = None
    override_timeout_sec: float | None = None
    override_setup_timeout_sec: float | None = None
    max_timeout_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionConfig:
    agent: str
    model: dict
    settings: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return 'config-' + identity(self.to_dict())[:16]

    def to_dict(self) -> dict:
        from lexverse.models.verification_compat import convert_settings_names
        data = asdict(self)
        data["settings"] = convert_settings_names(self.settings, legacy=getattr(self, "_legacy", False))
        return data

    @classmethod
    def from_dict(cls, data, *, legacy=False):
        from lexverse.models.verification_compat import convert_settings_names
        value = cls(**{**data, "settings": convert_settings_names(data.get("settings", {}))})
        object.__setattr__(value, "_legacy", legacy)
        return value
