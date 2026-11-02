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

from lexverse.agents.external.acp_registry import (
    is_acp_registry_shorthand,
    registry_spec_from_agent_name,
)
from lexverse.models.agent.name import AgentName
from lexverse.utils.env import resolve_env_vars
from lexverse.utils.import_path import import_class

if TYPE_CHECKING:
    from lexverse.agents.base import BaseAgent
    from lexverse.models.trial.config import AgentConfig


def _import_agent_class(import_path: str) -> "type[BaseAgent]":
    from lexverse.agents.base import BaseAgent

    return cast(
        "type[BaseAgent]",
        import_class(import_path, base=BaseAgent, label="agent"),
    )


class AgentFactory:
    """
    Harbor-compatible general AgentFactory for LexVerse.
    """

    _AGENT_MAP: dict[AgentName, str] = {
        AgentName.VANILLA: "lexverse.agents.vanilla:VanillaAgent",
        AgentName.ORACLE: "lexverse.agents.oracle:OracleAgent",
        AgentName.NOP: "lexverse.agents.nop:NopAgent",
        AgentName.ACP: "lexverse.agents.external.acp:AcpAgent",
        AgentName.CLAUDE_CODE: "lexverse.agents.external.claude_code:ClaudeCode",
        AgentName.COPILOT_CLI: "lexverse.agents.external.copilot_cli:CopilotCli",
        AgentName.AIDER: "lexverse.agents.external.aider:Aider",
        AgentName.CLINE_CLI: "lexverse.agents.external.cline:ClineCli",
        AgentName.CODEX: "lexverse.agents.external.codex:Codex",
        AgentName.CURSOR_CLI: "lexverse.agents.external.cursor_cli:CursorCli",
        AgentName.GEMINI_CLI: "lexverse.agents.external.gemini_cli:GeminiCli",
        AgentName.ANTIGRAVITY_CLI: "lexverse.agents.external.antigravity_cli:AntigravityCli",
        AgentName.ROVODEV_CLI: "lexverse.agents.external.rovodev_cli:RovodevCli",
        AgentName.GOOSE: "lexverse.agents.external.goose:Goose",
        AgentName.GROK_BUILD: "lexverse.agents.external.grok_build:GrokBuild",
        AgentName.HERMES: "lexverse.agents.external.hermes:Hermes",
        AgentName.KIMI_CLI: "lexverse.agents.external.kimi_cli:KimiCli",
        AgentName.LANGGRAPH: "lexverse.agents.external.langgraph:LangGraph",
        AgentName.DEERFLOW: "lexverse.agents.external.deerflow:DeerFlow",
        AgentName.MINI_SWE_AGENT: "lexverse.agents.external.mini_swe_agent:MiniSweAgent",
        AgentName.NEMO_AGENT: "lexverse.agents.external.nemo_agent:NemoAgent",
        AgentName.SWE_AGENT: "lexverse.agents.external.swe_agent:SweAgent",
        AgentName.OPENCODE: "lexverse.agents.external.opencode:OpenCode",
        AgentName.MIMO: "lexverse.agents.external.mimo:MiMo",
        AgentName.OPENCLAW: "lexverse.agents.external.openclaw:OpenClaw",
        AgentName.OPENHANDS: "lexverse.agents.external.openhands:OpenHands",
        AgentName.OPENHANDS_SDK: "lexverse.agents.external.openhands_sdk:OpenHandsSDK",
        AgentName.PI: "lexverse.agents.external.pi:Pi",
        AgentName.QWEN_CODE: "lexverse.agents.external.qwen_code:QwenCode",
        AgentName.DEVIN: "lexverse.agents.external.devin:Devin",
        AgentName.TRAE_AGENT: "lexverse.agents.external.trae_agent:TraeAgent",
        AgentName.EVE: "lexverse.agents.external.eve:Eve",
        AgentName.VIBE: "lexverse.agents.external.vibe:Vibe",
    }

    @classmethod
    def supported_agent_names(cls) -> list[str]:
        return sorted(name.value for name in cls._AGENT_MAP)

    @classmethod
    def get_agent_class(cls, name: AgentName | str) -> "type[BaseAgent]":
        if isinstance(name, str) and is_acp_registry_shorthand(name):
            return _import_agent_class(cls._AGENT_MAP[AgentName.ACP])
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
        if isinstance(name, str) and is_acp_registry_shorthand(name):
            kwargs = {**kwargs, "registry_spec": registry_spec_from_agent_name(name)}
            agent_class = cls.get_agent_class(AgentName.ACP)
        else:
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
        if (
            import_path is None
            and name is not None
            and ":" in name
            and not is_acp_registry_shorthand(name)
        ):
            import_path, name = name, None

        agent_kwargs = {**config.kwargs, **kwargs}
        if config.override_setup_timeout_sec is not None:
            agent_kwargs["override_setup_timeout_sec"] = (
                config.override_setup_timeout_sec
            )
        if config.override_timeout_sec is not None:
            agent_kwargs.setdefault("timeout_sec", config.override_timeout_sec)
        if config.max_timeout_sec is not None:
            timeout = agent_kwargs.get("timeout_sec")
            agent_kwargs["timeout_sec"] = (min(timeout, config.max_timeout_sec)
                                           if timeout is not None else config.max_timeout_sec)

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



def agent_kind(name: str) -> str:
    """Resolve a name without importing heavyweight implementations."""
    if name in {"static", "native", "vanilla"}:
        return name
    if name in AgentFactory.supported_agent_names() or is_acp_registry_shorthand(name) or ":" in name:
        return "external"
    raise ValueError(f"Unknown agent: {name}")


def validate_agent(name, allowed, *, source, model=None, environment=None, network_mode=None):
    from lexverse.config import ConfigError
    try:
        kind = agent_kind(name)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if kind not in allowed:
        raise ConfigError(f"{source}: incompatible agent {name}; allowed kinds: {', '.join(sorted(allowed))}")
    if kind in {"vanilla", "external"} and model is not None and model.get("provider") != "openai_compatible":
        raise ConfigError(f"{name}: requires an OpenAI-compatible model service")
    if kind in {"vanilla", "external"} and environment is not None:
        from lexverse.models.task.environment import EnvironmentType
        if not isinstance(environment, str):
            raise ConfigError("environment must be a name or module:Class")
        if ":" not in environment:
            try:
                EnvironmentType(environment)
            except ValueError as exc:
                raise ConfigError(f"Unknown environment: {environment}") from exc
    if kind in {"vanilla", "external"} and network_mode == "allowlist":
        if environment != "docker":
            raise ConfigError("Network allowlists require Docker")
        if name not in {"vanilla", "codex", "claude-code", "oracle"}:
            raise ConfigError(f"{name}: network allowlist is unsupported")


def parse_agents(value: str) -> list[str]:
    from lexverse.config import ConfigError
    names = [name.strip() for name in value.split(",")]
    if not all(names) or len(set(names)) != len(names):
        raise ConfigError("--agents requires nonempty, distinct comma-separated names")
    for name in names:
        agent_kind(name)
    return names


# argparse does not reject repeated store options by itself.
import argparse


class AgentsSelection(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error("--agents may be supplied only once; use comma-separated names")
        try:
            names = parse_agents(values)
        except ValueError as exc:
            parser.error(str(exc))
        setattr(namespace, self.dest, names)
