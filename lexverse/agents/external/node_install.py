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

import os
import shlex

NVM_VERSION = "v0.40.2"
DEFAULT_NODE_MAJOR = 22


def nvm_node_install_snippet(node_major: int = DEFAULT_NODE_MAJOR) -> str:
    # Optional host-side mirrors for networks that cannot reach GitHub raw content or the npm registry.
    # LEXVERSE_NVM_SOURCE is a base URL serving nvm.sh for NVM_VERSION (e.g. a Gitee mirror).
    exports = "".join(
        f"export {name}={shlex.quote(value)} && "
        for name, value in (("NVM_NODEJS_ORG_MIRROR", os.environ.get("LEXVERSE_NODEJS_MIRROR")),
                            ("npm_config_registry", os.environ.get("LEXVERSE_NPM_REGISTRY")))
        if value
    )
    source = os.environ.get("LEXVERSE_NVM_SOURCE")
    fetch = (f'curl -fsSL {shlex.quote(source.rstrip("/") + "/nvm.sh")} -o "$NVM_DIR/nvm.sh"' if source else
             f"curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/{NVM_VERSION}/install.sh | PROFILE=/dev/null bash")
    # Reuse a Node.js the image already ships; HOME is a fresh per-trial mount, so nvm would reinstall every time.
    preinstalled = (f"command -v node >/dev/null 2>&1 && "
                    f"[ \"$(node -p 'process.versions.node.split(\".\")[0]')\" -ge {node_major} ]")
    return (
        exports
        + f"if {preinstalled}; then npm -v; else "
        + 'export NVM_DIR="$HOME/.nvm" && mkdir -p "$NVM_DIR" && '
        + fetch + " && "
        + '. "$NVM_DIR/nvm.sh" && '
        + f"nvm install {node_major} && npm -v; fi"
    )
