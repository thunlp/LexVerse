from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from . import UPSTREAM
from .environment import LegalWorldEnvironment
from .evaluation import LegalWorldAggregator, LegalWorldTaskVerifier
from .tasks import LegalWorldImporter, source_inputs, task_ranges
from .resources import law_resources, case_resources


class LegalWorldPlugin(BenchmarkPlugin):
    name = "legalworld"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    default_policy = "legalworld_native"
    default_environment = "legalworld_native"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_bundle(self, config):
        task_ranges(config.benchmark.get("tasks"))
        law_resources(config)
        case_resources(config)
        self.evaluation_context(config)
        if "simulation" not in config.models.roles:
            raise ConfigError("Legal-world requires models.roles.simulation for fixed native roles")
        return super().create_bundle(config)

    def create_importer(self, config=None):
        settings = config or {}
        return LegalWorldImporter(offline=settings.get("offline", False),
                                  task_definitions=settings.get("task_definitions", settings.get("tasks")))

    def task_inputs(self, config, task_types):
        return {**source_inputs(offline=True), **law_resources(config)[1], **case_resources(config)[1]}

    def evaluation_inputs(self, config):
        return {**source_inputs(offline=True), **law_resources(config)[1], **case_resources(config)[1]}

    def create_environment(self, config, provider):
        return LegalWorldEnvironment(config)

    def evaluation_context(self, config):
        if len(config.models.evaluators) != 1:
            raise ConfigError("Legal-world requires one model in models.evaluators for independent scoring")
        return {"judge_models": {"judge": config.models.evaluators[0]}}

    def create_verifier(self, config=None):
        return LegalWorldTaskVerifier(offline=(config or {}).get("offline", False))

    def create_aggregator(self):
        return LegalWorldAggregator()


PLUGIN = LegalWorldPlugin()
