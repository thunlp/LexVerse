from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool

from lexverse.config import ConfigError
from lexverse.tasks.user import AgentLimits, RetrievalConfig, StrictModel, TaskCapabilities


class Resources(StrictModel):
    knowledge: list[Path] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    mcp: list[str] = Field(default_factory=list)
    tools: list[Literal["search_knowledge", "fetch_knowledge"]] = Field(
        default_factory=lambda: ["search_knowledge", "fetch_knowledge"])


class Override(Resources):
    limits: dict = Field(default_factory=dict)


class BenchmarkCapabilities(StrictModel):
    enabled: StrictBool = False
    registry: Path | None = None
    defaults: Resources = Field(default_factory=Resources)
    task_overrides: dict[str, Override] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    limits: AgentLimits = Field(default_factory=AgentLimits)
    context_window_tokens: int | None = Field(default=None, gt=0, strict=True)

    def effective(self, kind):
        resources = self.defaults.model_dump()
        limits = self.limits.model_dump()
        override = self.task_overrides.get(kind)
        if override:
            fields = override.model_dump(exclude_unset=True)
            limits.update(fields.pop("limits", {}))
            resources.update(fields)
        return TaskCapabilities(registry=self.registry, **resources), AgentLimits.model_validate(limits)


def parse_capabilities(value, benchmark, model, base):
    try:
        settings = BenchmarkCapabilities.model_validate(value)
        if not settings.enabled:
            return settings.model_dump(mode="json")
        if benchmark not in {"lawbench", "lexeval"}:
            raise ValueError("Capability enhancement supports lawbench and lexeval only")
        if model.provider != "openai_compatible":
            raise ValueError("Capability enhancement requires an OpenAI-compatible service")
        import yaml
        catalog = Path(__file__).parents[1] / "benchmarks" / benchmark / "catalog.yaml"
        known = yaml.safe_load(catalog.read_text())["tasks"]
        if set(settings.task_overrides) - known.keys():
            raise ValueError("Unknown capability task override")
        def resolve(path):
            path = path.expanduser()
            return (path if path.is_absolute() else base / path).resolve()
        if settings.registry is not None:
            settings.registry = resolve(settings.registry)
        settings.retrieval.embedding.resolve_path(base)
        settings.defaults.knowledge = [resolve(path) for path in settings.defaults.knowledge]
        for selection in settings.task_overrides.values():
            if "knowledge" in selection.model_fields_set:
                selection.knowledge = [resolve(path) for path in selection.knowledge]
        if settings.retrieval.overlap_tokens >= settings.retrieval.chunk_tokens:
            raise ValueError("retrieval overlap must be smaller than chunk size")
        from lexverse.capabilities.registry import load_registry
        registry = load_registry(settings.registry)
        for kind in known:
            selection, _ = settings.effective(kind)
            for source in selection.knowledge:
                if not source.is_dir():
                    raise ValueError(f"Knowledge directory does not exist: {source}")
            for skill in selection.skills:
                if skill not in registry.skills:
                    raise ValueError(f"Unknown skill: {skill}")
                if registry.skills[skill].requires_confirmation:
                    raise ValueError(f"Skill {skill} requires human fact confirmation")
            if set(selection.mcp) - registry.mcp.keys():
                raise ValueError("Unknown MCP service")
        normalized = settings.model_dump(mode="json")
        normalized["task_overrides"] = {kind: override.model_dump(mode="json", exclude_unset=True)
                                        for kind, override in settings.task_overrides.items()}
        return normalized
    except (ValueError, OSError) as exc:
        raise ConfigError(f"generation.capabilities: {exc}") from exc
