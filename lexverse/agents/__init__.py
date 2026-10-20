import importlib

_EXPORTS = {
    "AgentFactory": "lexverse.agents.factory:AgentFactory",
    "AgentSetupInfo": "lexverse.agents.base:AgentSetupInfo",
    "BaseAgent": "lexverse.agents.base:BaseAgent",
    "BaseInstalledAgent": "lexverse.agents.installed.base:BaseInstalledAgent",
    "CliFlag": "lexverse.agents.installed.base:CliFlag",
    "EnvVar": "lexverse.agents.installed.base:EnvVar",
    "LexVerseAgent": "lexverse.agents.lexverse_agent:LexVerseAgent",
    "NonZeroAgentExitCodeError": "lexverse.agents.installed.base:NonZeroAgentExitCodeError",
    "NopAgent": "lexverse.agents.nop:NopAgent",
    "OracleAgent": "lexverse.agents.oracle:OracleAgent",
    "with_prompt_template": "lexverse.agents.installed.base:with_prompt_template",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod, attr = _EXPORTS[name].split(":", 1)
    return getattr(importlib.import_module(mod), attr)
