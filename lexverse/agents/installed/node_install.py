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

"""Shared shell snippet for installing Node.js via nvm in agent environments."""

NVM_VERSION = "v0.40.2"
DEFAULT_NODE_MAJOR = 22


def nvm_node_install_snippet(node_major: int = DEFAULT_NODE_MAJOR) -> str:
    return (
        f"curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/{NVM_VERSION}/install.sh | bash && "
        'export NVM_DIR="$HOME/.nvm" && '
        '\\. "$NVM_DIR/nvm.sh" || true && '
        "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
        f"nvm install {node_major} && npm -v"
    )
