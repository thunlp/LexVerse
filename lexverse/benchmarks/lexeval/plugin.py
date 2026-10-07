from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.environments import DirectResponseEnvironment
from lexverse.interaction.participants import ModelParticipant
from lexverse.interaction.policies import DirectResponsePolicy
from lexverse.benchmarks.lexeval import UPSTREAM as LEXEVAL

from .evaluation import LexEvalAggregator, LexEvalTaskVerifier
from .tasks import LexEvalImporter, LexEvalLoader


class LexEvalPlugin(BenchmarkPlugin):
    name = "lexeval"
    upstream = LEXEVAL
    source_version = LEXEVAL.commit
    default_policy = "direct_response"
    default_environment = "direct_response"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        config = config or {}
        return LexEvalImporter(
            offline=config.get("offline", False), catalog=self.load_catalog(),
            few_shot_path=Path(config["few_shot_path"]) if config.get("few_shot_path") else None,
        )

    def task_inputs(self, config, task_types):
        loader = LexEvalLoader(offline=True)
        paths = {kind: loader.task_file(kind) for kind in task_types}
        if config.benchmark.get("few_shot_path"):
            paths["few_shot"] = Path(config.benchmark["few_shot_path"])
        return paths

    def create_environment(self, config, provider):
        return DirectResponseEnvironment(
            policy=DirectResponsePolicy(),
            participants={"assistant": ModelParticipant("assistant", provider)},
        )

    def create_verifier(self, config=None):
        config = config or {}
        return LexEvalTaskVerifier(offline=config.get("offline", False))

    def create_aggregator(self):
        return LexEvalAggregator()

PLUGIN = LexEvalPlugin()
