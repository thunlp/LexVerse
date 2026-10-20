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

from __future__ import annotations

import importlib
from typing import Any

IMPORT_PATH_FORMAT = "module.path:ClassName"


def import_symbol(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Import path must be in format '{IMPORT_PATH_FORMAT}'")

    module_path, symbol_name = import_path.split(":", 1)
    if not module_path or not symbol_name:
        raise ValueError(f"Import path must be in format '{IMPORT_PATH_FORMAT}'")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"Failed to import module '{module_path}': {exc}") from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module '{module_path}' has no class '{symbol_name}'"
        ) from exc


def import_class(
    import_path: str,
    *,
    base: type | None = None,
    label: str = "symbol",
) -> type:
    symbol = import_symbol(import_path)
    if not isinstance(symbol, type):
        raise TypeError(f"Imported {label} '{import_path}' must be a class")
    if base is not None and not issubclass(symbol, base):
        raise TypeError(
            f"Imported {label} '{import_path}' must subclass {base.__name__}"
        )
    return symbol
