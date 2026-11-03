from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from lexverse.config import ConfigError, parse_model_config, resolve_path


class AgentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    model: dict | None = None
    override_timeout_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class EnvironmentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["docker", "local"] = "docker"
    override_cpus: int | None = Field(default=None, gt=0)
    override_memory_mb: int | None = Field(default=None, gt=0)


class VerifierSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    override_timeout_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class JobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    _resolved_models: dict = PrivateAttr(default_factory=dict)
    _resolved_resources: object = PrivateAttr(default=None)
    _resolved_retrieval: object = PrivateAttr(default=None)

    schema_version: Literal["1.0"] = "1.0"
    path: Path | None = None
    benchmark: dict | None = None
    models: dict
    agents: list[AgentSelection] = Field(min_length=1)
    environment: EnvironmentSelection = Field(default_factory=EnvironmentSelection)
    verifier: VerifierSelection = Field(default_factory=VerifierSelection)
    resources: dict = Field(default_factory=dict)
    attempts: int = Field(default=1, gt=0, strict=True)
    concurrency: int = Field(default=1, gt=0, strict=True)
    max_retries: int = Field(default=0, ge=0, strict=True)
    generation: dict = Field(default_factory=dict)
    verification: dict = Field(default_factory=dict)
    output_dir: Path | None = None
    interactive: bool = False
    export_excel: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_verification(cls, value):
        from lexverse.models.verification_compat import convert_config_names
        return convert_config_names(value) if isinstance(value, dict) else value

    @model_validator(mode="before")
    @classmethod
    def reject_inline_credentials(cls, value):
        def check(item):
            if isinstance(item, dict):
                if any(str(key).lower() in {"api_key", "openai_api_key", "token", "hf_token",
                                                "authorization", "password", "headers"} for key in item):
                    raise ValueError("credentials belong in secrets profiles, not Job configuration")
                for child in item.values():
                    check(child)
            elif isinstance(item, list):
                for child in item:
                    check(child)
        check(value)
        return value

    @model_validator(mode="after")
    def validate_experiment(self):
        if (self.path is None) == (self.benchmark is None):
            raise ValueError("Select exactly one task path or benchmark")
        if self.benchmark is not None and not self.benchmark.get("name"):
            raise ValueError("benchmark.name is required")
        if self.benchmark and self.benchmark.get("name") == "j1bench":
            if set(self.benchmark) & {"scenario", "scenarios", "case_database"}:
                raise ValueError("Use benchmark.tasks and benchmark.case_databases for J1Bench")
        if "limit" in self.generation:
            raise ValueError("Use benchmark.default_limit or benchmark.tasks[].limit")
        if set(self.models) - {"default", "roles", "summary", "verifiers"}:
            raise ValueError("Unsupported models field")
        if "default" not in self.models:
            raise ValueError("models.default is required")
        names = [agent.name for agent in self.agents]
        if any(not name.strip() or "," in name for name in names):
            raise ValueError("agents must contain individual nonempty names")
        if len({agent.model_dump_json() for agent in self.agents}) != len(self.agents):
            raise ValueError("Duplicate Agent configurations")
        removed = {"capabilities", "n_attempts", "max_attempts", "concurrency", "resume"}
        if removed.intersection(self.generation):
            raise ValueError("Resources and scheduling belong in Job fields, not generation")
        return self

    def resolve(self, base: Path):
        if self.benchmark:
            for name in ("few_shot_path", "data_dir", "case_database", "submission_path", "candidate_pool_path", "decomposed_query_path", "decomposed_doc_path"):
                if self.benchmark.get(name):
                    self.benchmark[name] = resolve_path(self.benchmark[name], base)
            if isinstance(self.benchmark.get("case_databases"), dict):
                self.benchmark["case_databases"] = {
                    name: resolve_path(path, base) for name, path in self.benchmark["case_databases"].items()
                }
        for field in ("path", "output_dir"):
            value = getattr(self, field)
            if value is not None:
                value = value.expanduser()
                setattr(self, field, (value if value.is_absolute() else base / value).resolve())
        values = [("default", self.models["default"])]
        roles = self.models.get("roles", {})
        verifiers = self.models.get("verifiers", [])
        if not isinstance(roles, dict) or not isinstance(verifiers, list):
            raise ConfigError("models.roles must be a mapping and models.verifiers a list")
        values.extend((f"roles.{key}", value) for key, value in roles.items())
        values.extend((f"verifiers.{index}", value) for index, value in enumerate(verifiers))
        if self.models.get("summary") is not None:
            values.append(("summary", self.models["summary"]))
        values.extend((f"agents.{index}.model", agent.model) for index, agent in enumerate(self.agents)
                      if agent.model is not None)
        self._resolved_models = {name: parse_model_config(value, field=f"models.{name}", base_dir=base)
                                 for name, value in values}
        from lexverse.capabilities.resources import ResourceSelection
        self._resolved_resources = ResourceSelection.model_validate(self.resources).resolve(base)
        self.resources = self._resolved_resources.model_dump(mode="json", exclude_defaults=True)
        self._resolved_retrieval = None
        if "retrieval" in self.generation:
            from lexverse.models.task.config import RetrievalConfig
            retrieval = RetrievalConfig.model_validate(self.generation["retrieval"])
            retrieval.embedding.resolve_path(base)
            self._resolved_retrieval = retrieval
            self.generation["retrieval"] = retrieval.model_dump(mode="json")
        return self
