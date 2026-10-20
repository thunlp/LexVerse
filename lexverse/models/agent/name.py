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


class AgentName(str, Enum):
    LEXVERSE = "lexverse"
    ORACLE = "oracle"
    NOP = "nop"
    ACP = "acp"
    CLAUDE_CODE = "claude-code"
    CLINE_CLI = "cline-cli"
    AIDER = "aider"
    CODEX = "codex"
    CURSOR_CLI = "cursor-cli"
    GEMINI_CLI = "gemini-cli"
    ANTIGRAVITY_CLI = "antigravity-cli"
    ROVODEV_CLI = "rovodev-cli"
    GOOSE = "goose"
    GROK_BUILD = "grok-build"
    HERMES = "hermes"
    MINI_SWE_AGENT = "mini-swe-agent"
    NEMO_AGENT = "nemo-agent"
    SWE_AGENT = "swe-agent"
    OPENCODE = "opencode"
    OPENCLAW = "openclaw"
    OPENHANDS = "openhands"
    OPENHANDS_SDK = "openhands-sdk"
    KIMI_CLI = "kimi-cli"
    LANGGRAPH = "langgraph"
    DEERFLOW = "deerflow"
    MIMO = "mimo"
    PI = "pi"
    QWEN_CODE = "qwen-coder"
    COPILOT_CLI = "copilot-cli"
    DEVIN = "devin"
    TRAE_AGENT = "trae-agent"
    EVE = "eve"
    VIBE = "vibe"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
