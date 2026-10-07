"""Load benchmark configuration and provider credentials."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
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
    base_url: str | None
    api_key: str


@dataclass
class ModelConfig:
    provider: str
    name: str
    profile: str | None = None
    load_parameters: dict[str, Any] = field(default_factory=dict)
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelsConfig:
    default: ModelConfig
    roles: dict[str, ModelConfig] = field(default_factory=dict)
    summary: ModelConfig | None = None
    evaluators: list[ModelConfig] = field(default_factory=list)


@dataclass
class ResolvedConfig:
    schema_version: float
    benchmark: dict[str, Any]
    models: ModelsConfig
    generation: dict[str, Any]
    evaluation: dict[str, Any]
    raw: dict[str, Any]
    source_path: Path

    def hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def is_local_path(name: str) -> bool:
    return name.startswith(("/", "./", "../", "~/"))


def resolve_path(name: str, base_dir: Path | None) -> str:
    path = Path(name).expanduser()
    return str((path if path.is_absolute() else (base_dir or Path.cwd()) / path).resolve())


def validate_local_generation(options: dict, *, field: str = "generation_parameters") -> None:
    if set(options) - {"temperature", "max_tokens", "top_p"}:
        raise ConfigError(f"{field}: local generation supports temperature, max_tokens and top_p only")
    for key, default in (("temperature", 0), ("top_p", 1)):
        value = options.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 <= value if key == "temperature" else 0 < value <= 1):
            raise ConfigError(f"{field}.{key} is out of range")
    tokens = options.get("max_tokens", 512)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ConfigError(f"{field}.max_tokens must be a positive integer")


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


def parse_model_config(value: object, *, field: str, base_dir: Path | None = None) -> ModelConfig:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    unknown = set(value) - {"provider", "name", "profile", "load_parameters", "generation_parameters", "input"}
    if unknown:
        raise ConfigError(f"{field}: unsupported fields {sorted(unknown)}; use the new model schema")
    provider = value.get("provider")
    if not isinstance(provider, str) or provider not in {"openai_compatible", "transformers", "vllm"}:
        raise ConfigError(f"{field}.provider must be openai_compatible, transformers or vllm")
    profile = value.get("profile")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{field}.name is required")
    for key in ("load_parameters", "generation_parameters", "input"):
        if key in value and not isinstance(value[key], dict):
            raise ConfigError(f"{field}.{key} must be a mapping")
    loading = value.get("load_parameters", {})
    generation = value.get("generation_parameters", {})
    input_config = value.get("input", {})
    if provider == "openai_compatible":
        if not isinstance(profile, str) or not profile.strip():
            raise ConfigError(f"{field}.profile is required for API models")
        if "load_parameters" in value or "input" in value:
            raise ConfigError(f"{field}: load_parameters and input are only for local providers")
    else:
        if "profile" in value:
            raise ConfigError(f"{field}: local providers do not accept profile")
        shared = {"revision", "cache_dir", "local_files_only", "trust_remote_code", "dtype"}
        allowed = shared | ({"device_map"} if provider == "transformers" else {
            "tensor_parallel_size", "gpu_memory_utilization", "max_model_len",
        })
        if set(loading) - allowed:
            raise ConfigError(f"{field}.load_parameters: unsupported fields for {provider}: {sorted(set(loading) - allowed)}")
        for key in ("local_files_only", "trust_remote_code"):
            if key in loading and not isinstance(loading[key], bool):
                raise ConfigError(f"{field}.load_parameters.{key} must be boolean")
        if not isinstance(loading.get("dtype", "auto"), str) or loading.get("dtype", "auto") not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigError(f"{field}.load_parameters.dtype is unsupported")
        if provider == "transformers" and (not isinstance(loading.get("device_map", "auto"), str) or loading.get("device_map", "auto") not in {"auto", "cpu", "mps", "cuda"}):
            raise ConfigError(f"{field}.load_parameters.device_map must be auto, cpu, mps or cuda")
        for key in ("tensor_parallel_size", "max_model_len"):
            if key in loading and (isinstance(loading[key], bool) or not isinstance(loading[key], int) or loading[key] <= 0):
                raise ConfigError(f"{field}.load_parameters.{key} must be a positive integer")
        if "gpu_memory_utilization" in loading and (
            isinstance(loading["gpu_memory_utilization"], bool) or not isinstance(loading["gpu_memory_utilization"], (int, float))
            or not 0 < loading["gpu_memory_utilization"] <= 1
        ):
            raise ConfigError(f"{field}.load_parameters.gpu_memory_utilization must be in (0, 1]")
        if is_local_path(name):
            name = resolve_path(name, base_dir)
            value["name"] = name
            if "revision" in loading:
                raise ConfigError(f"{field}: revision is only valid for HF models")
        elif not re.fullmatch(r"[\w.-]+/[\w.-]+", name):
            raise ConfigError(f"{field}.name must be an explicit local path or namespace/model HF ID")
        for key in ("revision", "cache_dir"):
            if key in loading and (not isinstance(loading[key], str) or not loading[key]):
                raise ConfigError(f"{field}.load_parameters.{key} must be a nonempty string")
        if "cache_dir" in loading:
            loading["cache_dir"] = resolve_path(loading["cache_dir"], base_dir)
        if set(input_config) - {"format", "chat_template", "chat_template_kwargs"}:
            raise ConfigError(f"{field}.input contains unsupported fields")
        if not isinstance(input_config.get("format", "chat"), str) or input_config.get("format", "chat") not in {"chat", "text"}:
            raise ConfigError(f"{field}.input.format must be chat or text")
        if not isinstance(input_config.get("chat_template_kwargs", {}), dict):
            raise ConfigError(f"{field}.input.chat_template_kwargs must be a mapping")
        if input_config.get("format") == "text" and set(input_config) - {"format"}:
            raise ConfigError(f"{field}: text input does not accept chat template options")
        if "chat_template" in input_config:
            if not isinstance(input_config["chat_template"], str) or not input_config["chat_template"]:
                raise ConfigError(f"{field}.input.chat_template must be a template file path")
            input_config["chat_template"] = resolve_path(input_config["chat_template"], base_dir)
        validate_local_generation(generation, field=field)
    return ModelConfig(provider=provider, profile=profile, name=name,
                       load_parameters=loading, generation_parameters=generation, input=input_config)


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
        if set(profile) - {"api_key", "base_url"}:
            raise SecretsMissing(
                f"profile {name!r} in {path}: only base_url and api_key are allowed; "
                "put provider and model name in the new model configuration"
            )
        if not isinstance(profile.get("api_key"), str) or not profile["api_key"]:
            raise SecretsMissing(f"profile {name!r} requires a nonempty api_key")
        if profile.get("base_url") is not None and not isinstance(profile["base_url"], str):
            raise SecretsMissing(f"profile {name!r}.base_url must be a URL string")
        return ProviderProfile(
            name=name,
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
        base_url=os.environ.get(f"LEXVERSE_{upper_name}_BASE_URL"),
        api_key=api_key,
    )


def profile_available(name: str) -> bool:
    try:
        load_profile(name)
        return True
    except SecretsMissing:
        return False


def parse_config(data: dict, *, source_path: Path) -> ResolvedConfig:
    def reject_inline_keys(value):
        if isinstance(value, dict):
            if set(value) & {"api_key", "token", "hf_token"}:
                raise ConfigError("credentials belong in secrets profiles or HF_TOKEN, not the saved run configuration")
            for item in value.values():
                reject_inline_keys(item)
        elif isinstance(value, list):
            for item in value:
                reject_inline_keys(item)
    reject_inline_keys(data)
    schema = data.get("schema_version", 1)
    modern = schema == 2 or (isinstance(schema, float) and schema == 1.0)
    if schema not in (1, 2):
        raise ConfigError(f"unsupported schema_version: {schema}")
    if modern:
        unknown = set(data) - {"schema_version", "benchmark", "models", "generation", "evaluation"}
        if unknown:
            raise ConfigError(f"schema_version 1.0: unsupported fields {sorted(unknown)}")
    elif set(data) & {"models", "generation"}:
        raise ConfigError("models and generation require schema_version: 1.0")

    bench = data.get("benchmark")
    if not isinstance(bench, dict) or "name" not in bench:
        raise ConfigError("benchmark.name is required")
    if bench["name"] not in {"lexeval", "j1bench", "lawbench", "plawbench", "dlawbench", "legalworld"}:
        raise ConfigError(f"unknown benchmark: {bench['name']!r}")
    for name in ("case_database", "few_shot_path"):
        if bench.get(name):
            bench[name] = str(Path(bench[name]).resolve())
    if isinstance(bench.get("case_databases"), dict):
        bench["case_databases"] = {
            name: str(Path(path).resolve()) for name, path in bench["case_databases"].items()
        }

    generation_key = "generation" if modern else "execution"
    generation = data.get(generation_key, {}) or {}
    evaluation = data.get("evaluation", {}) or {}
    if not isinstance(generation, dict) or not isinstance(evaluation, dict):
        raise ConfigError(f"{generation_key} and evaluation must be mappings")
    if "resume" in generation:
        raise ConfigError(f"{generation_key}.resume was removed; use run --resume RUN_DIR")
    if modern:
        models = data.get("models")
        if not isinstance(models, dict):
            raise ConfigError("models must be a mapping")
        unknown = set(models) - {"default", "roles", "summary", "evaluators"}
        if unknown:
            raise ConfigError(f"models: unsupported fields {sorted(unknown)}")
        if set(evaluation) & {"model", "judges"} or "summary_model" in bench:
            raise ConfigError("put all model definitions in models in schema_version 1.0")
        if bench["name"] == "lexeval" and set(evaluation) & {"choice_metric", "generation_metric"}:
            raise ConfigError("LexEval metrics are defined by the native catalog")
    else:
        # Legacy snapshots retain their raw layout and hash; only the resolved view changes.
        roles = data.get("roles", {}) or {}
        if not isinstance(roles, dict):
            raise ConfigError("roles must be a mapping")
        for role, value in roles.items():
            if not isinstance(value, dict) or set(value) - {"model"}:
                raise ConfigError(f"roles.{role}: use a complete model mapping; put generation parameters inside it")
        if "model" in evaluation and evaluation.get("judges"):
            raise ConfigError("evaluation.model and evaluation.judges cannot be combined")
        judges = evaluation.get("judges", [])
        if not isinstance(judges, list):
            raise ConfigError("evaluation.judges must be a list")
        models = {
            "default": data.get("model"),
            "roles": {role: value["model"] for role, value in roles.items() if "model" in value},
            "evaluators": [evaluation["model"]] if "model" in evaluation else judges,
        }
        if "summary_model" in bench:
            models["summary"] = bench["summary_model"]
        evaluation = {key: value for key, value in evaluation.items() if key not in {"model", "judges"}}
    model = parse_model_config(
        models.get("default"), field="models.default" if modern else "model", base_dir=source_path.parent,
    )
    roles = models.get("roles", {}) or {}
    if not isinstance(roles, dict):
        raise ConfigError("models.roles must be a mapping")
    if roles and bench["name"] not in {"j1bench", "dlawbench", "legalworld"}:
        raise ConfigError("roles model overrides are currently supported only by J1Bench, DLawBench and Legal-world")
    role_models = {role: parse_model_config(value, field=f"models.roles.{role}", base_dir=source_path.parent)
                   for role, value in roles.items()}
    evaluators = models.get("evaluators", [])
    if not isinstance(evaluators, list):
        raise ConfigError("models.evaluators must be a list of model mappings")
    evaluator_models = [parse_model_config(value, field=f"models.evaluators.{index}", base_dir=source_path.parent)
                        for index, value in enumerate(evaluators)]
    if bench["name"] != "dlawbench" and len(evaluator_models) > 1:
        raise ConfigError("this benchmark accepts one model in models.evaluators")
    if modern and bench["name"] in {"lexeval", "lawbench"} and evaluator_models:
        raise ConfigError("this benchmark uses its native scorer without models.evaluators")
    summary_model = None
    if "summary" in models and bench["name"] != "j1bench":
        raise ConfigError("models.summary is supported only by J1Bench KQ/LC")
    if bench["name"] == "j1bench":
        # Upstream KQ/LC summaries use this fixed model independently of role models.
        summary = models.setdefault("summary", {
            "provider": "openai_compatible", "profile": model.profile or "openai",
            "name": "gpt-4o-2024-08-06",
            "generation_parameters": {"temperature": 0, "max_tokens": 4096},
        })
        summary_model = parse_model_config(
            summary, field="models.summary", base_dir=source_path.parent,
        )
    for scope in (generation, evaluation):
        for key in ("timeout_sec", "model_startup_timeout_sec", "model_request_timeout_sec"):
            if key in scope and (isinstance(scope[key], bool) or not isinstance(scope[key], (int, float)) or not math.isfinite(scope[key]) or scope[key] <= 0):
                raise ConfigError(f"{key} must be positive")
    from lexverse.tasks.bundle import TaskSelection
    try:
        selection = TaskSelection.from_config(bench, generation)
        if bench["name"] == "legalworld":
            from lexverse.benchmarks.legalworld.tasks import task_ranges
            if "start_stage" in bench or "end_stage" in bench:
                raise ConfigError("put Legal-world start_stage and end_stage inside each benchmark.tasks entry")
            task_ranges(bench.get("tasks"))
        if roles and bench["name"] == "dlawbench" and set(roles) - {"lawyer", "client"}:
            raise ConfigError("DLawBench roles must be lawyer or client")
        if roles and bench["name"] == "legalworld" and set(roles) - {"lawyer", "simulation"}:
            raise ConfigError("Legal-world roles must be lawyer or simulation")
        if roles and bench["name"] == "j1bench":
            from lexverse.benchmarks.j1bench.scenarios import SCENARIOS
            names = selection.select_types(list(SCENARIOS))
            supported_roles = {role for name in names for role in SCENARIOS[name].roles}
            if set(roles) - supported_roles:
                raise ConfigError(f"roles not present in the selected scenarios: {sorted(set(roles) - supported_roles)}")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    return ResolvedConfig(
        schema_version=1.0,
        benchmark=bench,
        generation=generation,
        evaluation=evaluation,
        models=ModelsConfig(default=model, roles=role_models, summary=summary_model, evaluators=evaluator_models),
        raw=data,
        source_path=source_path,
    )


def load_config(path: str | Path) -> ResolvedConfig:
    import yaml

    p = Path(path).resolve()
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    return parse_config(data, source_path=p)
