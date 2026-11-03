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

"""
Utility functions for template rendering with Jinja2 support.
"""

from pathlib import Path

from jinja2 import (
    Environment,
    StrictUndefined,
    TemplateSyntaxError,
    UndefinedError,
    meta,
)


def render_prompt_template(template_path: Path, instruction: str) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    template_content = template_path.read_text(encoding="utf-8")

    env_for_parsing = Environment()
    try:
        ast = env_for_parsing.parse(template_content)
        undeclared_variables = meta.find_undeclared_variables(ast)

        if "instruction" not in undeclared_variables:
            raise ValueError(
                f"Prompt template {template_path} must include an 'instruction' "
                f"variable. Use {{{{ instruction }}}} (jinja2 syntax) in your template."
            )
    except TemplateSyntaxError:
        if (
            "{{ instruction }}" not in template_content
            and "{{instruction}}" not in template_content
        ):
            raise ValueError(
                f"Prompt template {template_path} must include an 'instruction' "
                f"variable. Use {{{{ instruction }}}} (jinja2 syntax) in your template."
            )

    try:
        env = Environment(undefined=StrictUndefined)
        template = env.from_string(template_content)
        return template.render(instruction=instruction)
    except UndefinedError as e:
        raise ValueError(
            f"Prompt template {template_path} has undefined variables. Error: {e}"
        ) from e
    except Exception as e:
        raise ValueError(f"Error rendering prompt template {template_path}: {e}") from e
