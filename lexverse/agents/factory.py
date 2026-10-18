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

from pathlib import Path
from typing import TYPE_CHECKING, cast

from lexverse.models.agent.name import AgentName
from lexverse.utils.env import resolve_env_vars
from lexverse.utils.import_path import import_class

if TYPE_CHECKING:
    from lexverse.agents.base import BaseAgent
    from lexverse.models.trial import AgentConfig


def _import_agent_class(import_path: str) -> "type[BaseAgent]":
    from lexverse.agents.base import BaseAgent

    return cast(
        "type[BaseAgent]",
        import_class(import_path, base=BaseAgent, label="agent"),
    )


class AgentFactory:
    """
    Harbor-compatible general AgentFactory for LexVerse.

    Supports both the built-in LexVerse LangGraph/DeepAgents implementation
    (`lexverse` / `deepagents`) and external CLI / framework agents (`claude-code`,
    `codex`, `qwen-code`, `gemini-cli`, `openhands`, `opencode`, `aider`,
    `kimi-cli`, `goose`, `cursor-cli`, `mini-swe-agent`, `pi`, `langgraph`,
    `oracle`, `nop`), as well as custom `module.path:ClassName` agents.
    """

    _AGENT_MAP: dict[AgentName, str] = {
        AgentName.LEXVERSE: "lexverse.agents.lexverse_agent:LexVerseAgent",
        AgentName.DEEPAGENTS: "lexverse.agents.lexverse_agent:LexVerseAgent",
        AgentName.ORACLE: "lexverse.agents.oracle:OracleAgent",
        AgentName.NOP: "lexverse.agents.nop:NopAgent",
        AgentName.CLAUDE_CODE: "lexverse.agents.installed.claude_code:ClaudeCode",
        AgentName.AIDER: "lexverse.agents.installed.aider:Aider",
        AgentName.CODEX: "lexverse.agents.installed.codex:Codex",
        AgentName.CURSOR_CLI: "lexverse.agents.installed.cursor_cli:CursorCli",
        AgentName.GEMINI_CLI: "lexverse.agents.installed.gemini_cli:GeminiCli",
        AgentName.GOOSE: "lexverse.agents.installed.goose:Goose",
        AgentName.KIMI_CLI: "lexverse.agents.installed.kimi_cli:KimiCli",
        AgentName.LANGGRAPH: "lexverse.agents.installed.langgraph:LangGraph",
        AgentName.MINI_SWE_AGENT: "lexverse.agents.installed.mini_swe_agent:MiniSweAgent",
        AgentName.OPENCODE: "lexverse.agents.installed.opencode:OpenCode",
        AgentName.OPENHANDS: "lexverse.agents.installed.openhands:OpenHands",
        AgentName.PI: "lexverse.agents.installed.pi:Pi",
        AgentName.QWEN_CODE: "lexverse.agents.installed.qwen_code:QwenCode",
        AgentName.QWEN_CODER: "lexverse.agents.installed.qwen_code:QwenCode",
    }

    @classmethod
    def supported_agent_names(cls) -> list[str]:
        return sorted(name.value for name in cls._AGENT_MAP)

    @classmethod
    def get_agent_class(cls, name: AgentName | str) -> "type[BaseAgent]":
        if isinstance(name, str) and ":" in name:
            return _import_agent_class(name)
        try:
            agent_enum = AgentName(name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown agent type: {name}. Supported agents: {cls.supported_agent_names()}"
            ) from exc
        if agent_enum not in cls._AGENT_MAP:
            raise ValueError(
                f"Agent '{name}' is not registered in AgentFactory. Supported: {cls.supported_agent_names()}"
            )
        return _import_agent_class(cls._AGENT_MAP[agent_enum])

    @classmethod
    def create_agent_from_name(
        cls,
        name: AgentName | str,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs,
    ) -> "BaseAgent":
        agent_class = cls.get_agent_class(name)
        return agent_class(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @classmethod
    def create_agent_from_import_path(
        cls,
        import_path: str,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs,
    ) -> "BaseAgent":
        agent_class = _import_agent_class(import_path)
        return agent_class(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @classmethod
    def create_agent_from_config(
        cls,
        config: "AgentConfig",
        logs_dir: Path,
        **kwargs,
    ) -> "BaseAgent":
        extra_env = resolve_env_vars(config.env) if config.env else {}
        name = config.name
        import_path = config.import_path
        if import_path is None and name is not None and ":" in name:
            import_path, name = name, None

        agent_kwargs = {**config.kwargs, **kwargs}
        if config.override_setup_timeout_sec is not None:
            agent_kwargs["override_setup_timeout_sec"] = (
                config.override_setup_timeout_sec
            )
        if config.override_timeout_sec is not None:
            agent_kwargs.setdefault("timeout_sec", config.override_timeout_sec)

        if import_path is not None:
            return cls.create_agent_from_import_path(
                import_path,
                logs_dir=logs_dir,
                model_name=config.model_name,
                extra_env=extra_env,
                **agent_kwargs,
            )
        if name is not None:
            return cls.create_agent_from_name(
                name,
                logs_dir=logs_dir,
                model_name=config.model_name,
                extra_env=extra_env,
                **agent_kwargs,
            )
        raise ValueError(
            f"At least one of agent name or import path must be set. Supported: {cls.supported_agent_names()}"
        )


def build_agent(model, root: Path, *, tools=(), skills=None, checkpointer=None,
                middleware=(), interactive=False, system_prompt="", artifacts_root: Path | None = None,
                skills_root: Path | None = None):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemPermission
    from deepagents.profiles import (
        GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile,
    )
    from pydantic import ValidationError

    provider = model._get_ls_params().get("ls_provider")
    identifier = getattr(model, "model_name", None) or getattr(model, "model", None)
    if not provider:
        raise ValueError("Chat model must identify its provider for single-agent configuration")
    key = f"{provider}:{identifier}" if identifier else provider
    register_harness_profile(key, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"execute", "task", "delete"}),
    ))
    root = root.resolve()
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Agent filesystem must not contain symbolic links")
    backend = FilesystemBackend(root_dir=root, virtual_mode=True)
    if artifacts_root is not None or skills_root is not None:
        from deepagents.backends import CompositeBackend
        routes = {}
        if skills_root is not None:
            skills_root = skills_root.resolve()
            if any(path.is_symlink() for path in skills_root.rglob("*")):
                raise ValueError("Skills must not contain symbolic links")
            routes["/skills/"] = FilesystemBackend(root_dir=skills_root, virtual_mode=True)
        if artifacts_root is not None:
            artifacts_root = artifacts_root.resolve()
            if any(path.is_symlink() for path in artifacts_root.rglob("*")):
                raise ValueError("Artifacts must not contain symbolic links")
            routes["/artifacts/"] = FilesystemBackend(root_dir=artifacts_root, virtual_mode=True)
        backend = CompositeBackend(default=backend, routes=routes)
    permissions = [
        FilesystemPermission(operations=["read"], paths=["/inputs", "/inputs/**", "/skills", "/skills/**"]),
        FilesystemPermission(operations=["read", "write"], paths=[
            "/workspace", "/workspace/**", "/artifacts", "/artifacts/**",
            "/large_tool_results", "/large_tool_results/**",
            "/conversation_history", "/conversation_history/**", "/blobs", "/blobs/**",
        ]),
        FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
    ]
    user_tool = next((tool for tool in tools if getattr(tool, "name", None) == "ask_user"), None)

    def valid_question(request):
        try:
            arguments = user_tool.args_schema.model_validate(request.tool_call["args"])
            return bool(arguments.question.strip())
        except ValidationError:
            return False

    return create_deep_agent(
        model=model, backend=backend,
        tools=list(tools), skills=skills, middleware=list(middleware),
        permissions=permissions, checkpointer=checkpointer, system_prompt=system_prompt,
        interrupt_on={"ask_user": {"allowed_decisions": ["respond"], "when": valid_question}}
        if user_tool is not None else None,
    )
