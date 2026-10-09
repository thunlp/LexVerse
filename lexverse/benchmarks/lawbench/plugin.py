from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.environments import DirectResponseEnvironment
from lexverse.interaction.participants import ModelParticipant, AgentParticipant
from lexverse.interaction.policies import DirectResponsePolicy
from lexverse.benchmarks.lawbench import UPSTREAM as LAWBENCH

from .evaluation import LawBenchAggregator, LawBenchTaskVerifier
from .tasks import LawBenchImporter, LawBenchLoader


class LawBenchPlugin(BenchmarkPlugin):
    name = "lawbench"
    upstream = LAWBENCH
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

    def task_inputs(self, config, task_types):
        loader = LawBenchLoader(offline=True)
        return {kind: loader.task_file(kind, config.benchmark.get("shot", "zero_shot")) for kind in task_types}

    def create_environment(self, config, provider):
        from lexverse.runtime.enhancement import enabled
        if enabled(config):
            return DirectResponseEnvironment(
                policy=DirectResponsePolicy(),
                participants={"assistant": AgentParticipant("assistant", config)},
            )
        return DirectResponseEnvironment(
            policy=DirectResponsePolicy(),
            participants={"assistant": ModelParticipant("assistant", provider)},
        )

    def create_verifier(self, config=None):
        config = config or {}
        return LawBenchTaskVerifier(offline=config.get("offline", False))

    def create_aggregator(self):
        return LawBenchAggregator()

PLUGIN = LawBenchPlugin()
