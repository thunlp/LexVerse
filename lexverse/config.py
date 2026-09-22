"""Load benchmark configuration and provider credentials."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

_DEFAULT_SECRET_LOCATIONS = (
    Path.cwd() / "configs" / "secrets.local.yaml",
    Path.home() / ".lexverse" / "secrets.yaml",
)


class ConfigError(ValueError):
    pass


class SecretsMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    provider: str
    base_url: str | None
    api_key: str


@dataclass
class ModelConfig:
    profile: str
    name: str
    parameters: dict[str, Any]


@dataclass
class ResolvedConfig:
    schema_version: int
    benchmark: dict[str, Any]
    execution: dict[str, Any]
    evaluation: dict[str, Any]
    model: ModelConfig
    raw: dict[str, Any]
    source_path: Path

    def hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path) -> ResolvedConfig:
    import yaml

    p = Path(path).resolve()
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    schema = data.get("schema_version", 1)
    if schema != 1:
        raise ConfigError(f"unsupported schema_version: {schema}")

    bench = data.get("benchmark")
    if not isinstance(bench, dict) or "name" not in bench:
        raise ConfigError("benchmark.name is required")
    if bench["name"] not in {"lexeval", "j1bench", "lawbench"}:
        raise ConfigError(f"unknown benchmark: {bench['name']!r}")

    model = parse_model_config(data.get("model"), field="model")
    execution = data.get("execution", {}) or {}
    evaluation = data.get("evaluation", {}) or {}

    return ResolvedConfig(
        schema_version=schema,
        benchmark=bench,
        execution=execution,
        evaluation=evaluation,
        model=model,
        raw=data,
        source_path=p,
    )


def parse_model_config(value: object, *, field: str) -> ModelConfig:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    profile = value.get("profile")
    name = value.get("name")
    parameters = value.get("parameters", {}) or {}
    if not isinstance(profile, str) or not profile.strip():
        raise ConfigError(f"{field}.profile is required")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{field}.name is required")
    if not isinstance(parameters, dict):
        raise ConfigError(f"{field}.parameters must be a mapping")
    return ModelConfig(profile=profile, name=name, parameters=parameters)


def _expand_secret(value: object) -> object:
    if isinstance(value, str):
        def substitute(match: re.Match[str]) -> str:
            variable = match.group(1)
            resolved = os.environ.get(variable)
            if resolved is None:
                raise SecretsMissing(
                    f"env var {variable!r} referenced but not set"
                )
            return resolved
        return _ENV_VAR_RE.sub(substitute, value)
    return value


def load_profile(name: str, path: Path | None = None) -> ProviderProfile:
    """Load connection credentials without mixing in model selection.

    Search order: explicit path, ``LEXVERSE_SECRETS_FILE``, repository-local
    ``configs/secrets.local.yaml``, ``~/.lexverse/secrets.yaml``, then
    ``LEXVERSE_<PROFILE>_*`` environment variables.
    """
    if path is None:
        env_path = os.environ.get("LEXVERSE_SECRETS_FILE")
        candidates = [Path(env_path)] if env_path else []
        candidates += list(_DEFAULT_SECRET_LOCATIONS)
        path = next((candidate for candidate in candidates if candidate.exists()), None)

    if path is not None and path.exists():
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if name not in data:
            raise SecretsMissing(f"profile {name!r} not found in {path}")
        profile = data[name]
        if not isinstance(profile, dict):
            raise SecretsMissing(f"profile {name!r} in {path} must be a mapping")
        if "model" in profile:
            raise SecretsMissing(
                f"profile {name!r} in {path} contains deprecated 'model'; "
                "move it to model.name in the benchmark config"
            )
        return ProviderProfile(
            name=name,
            provider=str(profile.get("provider", "openai_compatible")),
            base_url=_expand_secret(profile.get("base_url")),  # type: ignore[arg-type]
            api_key=_expand_secret(profile["api_key"]),  # type: ignore[arg-type]
        )

    upper_name = name.upper()
    api_key = os.environ.get(f"LEXVERSE_{upper_name}_API_KEY")
    if not api_key:
        raise SecretsMissing(
            f"no secrets file found and LEXVERSE_{upper_name}_API_KEY not set; "
            "see configs/secrets.example.yaml"
        )
    return ProviderProfile(
        name=name,
        provider=os.environ.get(
            f"LEXVERSE_{upper_name}_PROVIDER", "openai_compatible"
        ),
        base_url=os.environ.get(f"LEXVERSE_{upper_name}_BASE_URL"),
        api_key=api_key,
    )


def profile_available(name: str) -> bool:
    try:
        load_profile(name)
        return True
    except SecretsMissing:
        return False
