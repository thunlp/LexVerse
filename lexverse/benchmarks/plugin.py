from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lexverse.agents.native import NativeRunner
from lexverse.tasks.bundle import TaskBundle, TaskSelection
from lexverse.models.task.task import LexVerseTask
from lexverse.verifiers.benchmark import BenchmarkVerifier
from lexverse.benchmarks.aggregation import BenchmarkAggregator
from lexverse.benchmarks.sources import UpstreamSource
from .catalog import load_catalog


class TaskImporter(ABC):
    @abstractmethod
    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]: ...


def load_upstream(catalog_path: Path) -> UpstreamSource:
    """Build a benchmark's shared source from its packaged catalog."""
    from lexverse.config import ConfigError
    try:
        source = load_catalog(catalog_path)["source"]
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return UpstreamSource(**source)


class BenchmarkPlugin(ABC):
    name: str
    source_version: str
    catalog_path: Path
    upstream: UpstreamSource
    adapter_version = "1.0"
    supported_agents = frozenset({"native"})
    default_agent = "native"

    def validate_model_config(self, bench, models, modern=True):
        from lexverse.config import ConfigError
        if models.roles:
            raise ConfigError("This benchmark does not support role model overrides")
        if len(models.verifiers) > 1:
            raise ConfigError("This benchmark accepts at most one verifier model")
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")

    def generation_models(self, config, bundle):
        return {"default": config.models.default}

    def tested_model(self, config):
        return config.models.default

    def summary_metadata(self, aggregate):
        return {}

    def create_bundle(self, config: Any) -> TaskBundle:
        benchmark = config.benchmark
        from lexverse.config import ConfigError
        selection = TaskSelection.from_config(benchmark, config.generation)
        if self.name == "lexeval" and not selection.task_types:
            raise ConfigError("lexeval config must list benchmark.tasks")
        try:
            if self.name != "legalworld":
                selection.select_types(self.task_types())
        except ValueError as exc:
            raise ConfigError(f"unknown {self.name} task types: {exc}") from exc
        importer_config = {
            **benchmark,
            "tasks": selection.task_types,
            "task_definitions": benchmark.get("tasks"),
            "offline": bool(config.generation.get("offline", False)),
        }
        try:
            tasks = self.create_importer(importer_config).import_tasks(selection)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        self.selection_counts = selection.counts
        return TaskBundle(
            benchmark=self.name,
            source_version=self.source_version,
            tasks=tasks,
        )

    def restore_bundle(self, config: Any, references: list[dict]) -> TaskBundle:
        from lexverse.config import ConfigError
        tasks = []
        task_types = list(dict.fromkeys(ref["task_type"] for ref in references))
        for task_type in task_types:
            selected = [ref for ref in references if ref["task_type"] == task_type]
            importer = self.create_importer({
                **config.benchmark, "tasks": [task_type], "offline": True,
                "task_definitions": config.benchmark.get("tasks"),
            })
            imported = importer.import_tasks(TaskSelection(
                task_types=[task_type], sample_ids=[ref["sample_id"] for ref in selected],
            ))
            by_id = {task.id: task for task in imported}
            for ref in selected:
                task = by_id.get(ref["id"])
                if task is None or task.source.sample_id != ref["sample_id"]:
                    raise ConfigError(f"selected task missing or changed: {ref['id']}")
                tasks.append(task)
        if len({task.id for task in tasks}) != len(references):
            raise ConfigError("duplicate selected task IDs in manifest")
        by_id = {task.id: task for task in tasks}
        return TaskBundle(benchmark=self.name, source_version=self.source_version,
                          tasks=[by_id[ref["id"]] for ref in references])

    @abstractmethod
    def create_importer(self, config: dict[str, Any] | None = None) -> TaskImporter: ...

    def task_inputs(self, config: Any, task_types: list[str]) -> dict[str, Path]:
        """Shared dataset files required to reconstruct selected tasks."""
        return {}

    def create_native_runner(self, config: Any) -> NativeRunner:
        raise ValueError(f"{self.name} has no native runner")

    def build_verification_context(self, config: Any) -> dict[str, Any]:
        return {}

    def resolve_verification_inputs(self, config: Any) -> dict[str, Path]:
        """External files used by the native evaluator outside the task bundle."""
        return {}

    @abstractmethod
    def create_verifier(self, config: dict[str, Any] | None = None) -> BenchmarkVerifier: ...

    @abstractmethod
    def create_aggregator(self) -> BenchmarkAggregator: ...

    def task_types(self) -> list[str]:
        return list(self.load_catalog()["tasks"])

    def load_catalog(self) -> dict[str, Any]:
        return load_catalog(self.catalog_path)
