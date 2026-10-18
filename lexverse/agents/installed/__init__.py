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

"""Harbor-compatible installed CLI agents for LexVerse."""

from lexverse.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    EnvVar,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)

__all__ = [
    "BaseInstalledAgent",
    "CliFlag",
    "EnvVar",
    "NonZeroAgentExitCodeError",
    "with_prompt_template",
]
