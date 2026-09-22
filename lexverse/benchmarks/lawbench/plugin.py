from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.environments import DirectResponseEnvironment
from lexverse.interaction.participants.model import ModelParticipant
from lexverse.interaction.policies.direct_response import DirectResponsePolicy
from lexverse.runtime.upstream import LAWBENCH

from .importer import LawBenchImporter
from .verifier import LawBenchTaskVerifier
from .aggregate import LawBenchAggregator


class LawBenchPlugin(BenchmarkPlugin):
    name = "lawbench"
    source_version = LAWBENCH.commit
    default_policy = "direct_response"
    default_environment = "direct_response"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        config = config or {}
        return LawBenchImporter(
            offline=config.get("offline", False), shot=config.get("shot", "zero_shot"),
            catalog=self.load_catalog(),
        )

    def create_verifier(self, config=None):
        config = config or {}
        return LawBenchTaskVerifier(offline=config.get("offline", False))

    def create_environment(self, config, provider):
        return DirectResponseEnvironment(
            policy=DirectResponsePolicy(),
            participants={"assistant": ModelParticipant("assistant", provider)},
        )

    def create_aggregator(self):
        return LawBenchAggregator()

PLUGIN = LawBenchPlugin()
