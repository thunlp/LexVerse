from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.benchmarks.lawbench import UPSTREAM as LAWBENCH

from .verifier import LawBenchAggregator, LawBenchVerifier
from .tasks import LawBenchImporter, LawBenchLoader


class LawBenchPlugin(BenchmarkPlugin):
    name = "lawbench"
    supported_agents = frozenset({"static", "vanilla", "external"})
    default_agent = "static"
    upstream = LAWBENCH
    source_version = LAWBENCH.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        config = config or {}
        return LawBenchImporter(
            offline=config.get("offline", False), shot=config.get("shot", "zero_shot"),
            catalog=self.load_catalog(),
        )

    def task_inputs(self, config, task_types):
        loader = LawBenchLoader(offline=True)
        return {kind: loader.task_file(kind, config.benchmark.get("shot", "zero_shot")) for kind in task_types}

    def create_verifier(self, config=None):
        config = config or {}
        return LawBenchVerifier(offline=config.get("offline", False))

    def create_aggregator(self):
        return LawBenchAggregator()

    def validate_model_config(self, bench, models, modern):
        from lexverse.config import ConfigError
        super().validate_model_config(bench, models, modern)
        if modern and models.verifiers:
            raise ConfigError("this benchmark uses its native scorer without models.verifiers")

PLUGIN = LawBenchPlugin()
