from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from lexverse.environments.base import ExecutionEnvironment
from lexverse.providers.base import Provider
from lexverse.tasks.bundle import TaskBundle
from lexverse.tasks.schema import LexVerseTask
from lexverse.tasks.selection import TaskSelection
from lexverse.verifiers.base import TaskVerifier
from lexverse.benchmarks.aggregate import BenchmarkAggregator


class TaskImporter(ABC):
    @abstractmethod
    def import_tasks(self, selection: TaskSelection) -> list[LexVerseTask]: ...


class BenchmarkPlugin(ABC):
    name: str
    source_version: str
    default_policy: str
    default_environment: str
    catalog_path: Path

    @abstractmethod
    def create_importer(self, config: dict[str, Any] | None = None) -> TaskImporter: ...

    @abstractmethod
    def create_verifier(self, config: dict[str, Any] | None = None) -> TaskVerifier: ...

    @abstractmethod
    def create_environment(
        self, config: Any, provider: Provider
    ) -> ExecutionEnvironment: ...

    @abstractmethod
    def create_aggregator(self) -> BenchmarkAggregator: ...

    def create_bundle(self, config: Any) -> TaskBundle:
        benchmark = config.benchmark
        task_types = benchmark.get("tasks")
        if self.name == "lexeval" and not task_types:
            from lexverse.config import ConfigError
            raise ConfigError("lexeval config must list benchmark.tasks")
        if task_types is not None:
            unknown = sorted(set(task_types) - set(self.task_types()))
            if unknown:
                from lexverse.config import ConfigError
                raise ConfigError(f"unknown {self.name} task types: {unknown}")
        selection = TaskSelection(
            task_types=task_types,
            sample_ids=benchmark.get("cases"),
            limit_per_task=config.execution.get("limit"),
        )
        importer_config = {
            **benchmark,
            "offline": bool(config.execution.get("offline", False)),
        }
        tasks = self.create_importer(importer_config).import_tasks(selection)
        return TaskBundle(
            benchmark=self.name,
            source_version=self.source_version,
            tasks=tasks,
        )

    def load_catalog(self) -> dict[str, Any]:
        data = yaml.safe_load(self.catalog_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), dict):
            raise ValueError(f"invalid benchmark catalog: {self.catalog_path}")
        return data

    def task_types(self) -> list[str]:
        return list(self.load_catalog()["tasks"])
