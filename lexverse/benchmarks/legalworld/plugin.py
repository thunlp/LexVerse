from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from . import UPSTREAM
from .environment import LegalWorldEnvironment
from .verifier import LegalWorldAggregator, LegalWorldVerifier
from .tasks import LegalWorldImporter, source_inputs, task_ranges
from .resources import law_resources, case_resources


class LegalWorldPlugin(BenchmarkPlugin):
    name = "legalworld"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_bundle(self, config):
        task_ranges(config.benchmark.get("tasks"))
        law_resources(config)
        case_resources(config)
        if "simulation" not in config.models.roles:
            raise ConfigError("Legal-world requires models.roles.simulation for fixed native roles")
        return super().create_bundle(config)

    def create_importer(self, config=None):
        settings = config or {}
        return LegalWorldImporter(offline=settings.get("offline", False),
                                  task_definitions=settings.get("task_definitions", settings.get("tasks")))

    def task_inputs(self, config, task_types):
        return {**source_inputs(offline=True), **law_resources(config)[1], **case_resources(config)[1]}

    def resolve_verification_inputs(self, config):
        return {**source_inputs(offline=True), **law_resources(config)[1], **case_resources(config)[1]}

    def create_native_runner(self, config):
        return LegalWorldEnvironment(config)

    def build_verification_context(self, config):
        if len(config.models.verifiers) != 1:
            raise ConfigError("Legal-world requires one model in models.verifiers for independent scoring")
        return {"judge_models": {"judge": config.models.verifiers[0]}}

    def create_verifier(self, config=None):
        return LegalWorldVerifier(offline=(config or {}).get("offline", False))

    def create_aggregator(self):
        return LegalWorldAggregator()


PLUGIN = LegalWorldPlugin()
