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

"""Utility functions for working with trajectory data."""

import json
import re
from typing import Any


def format_trajectory_json(data: dict[str, Any]) -> str:
    """Format trajectory JSON with compact numeric arrays on single lines."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    def compact_numeric_array(match):
        full_match = match.group(0)
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", full_match)
        if not numbers:
            return full_match
        return "[" + ", ".join(numbers) + "]"

    pattern = r"\[\s*\n\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*,\s*\n\s*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)*\s*\n\s*\]"
    json_str = re.sub(pattern, compact_numeric_array, json_str, flags=re.MULTILINE)
    return json_str
