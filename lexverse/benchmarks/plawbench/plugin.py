from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.environments import DirectResponseEnvironment
from lexverse.interaction.participants import ModelParticipant
from lexverse.interaction.policies import DirectResponsePolicy
from . import UPSTREAM
from .evaluation import PLawBenchAggregator, PLawBenchTaskVerifier
from .tasks import FILES, PLawBenchImporter


class PLawBenchPlugin(BenchmarkPlugin):
    name = "plawbench"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    default_policy = "direct_response"
    default_environment = "direct_response"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        return PLawBenchImporter(offline=(config or {}).get("offline", False))

    def task_inputs(self, config, task_types):
        importer = PLawBenchImporter(offline=True)
        return {filename: importer.files[filename] for kind in task_types for filename in FILES[kind].values()}

    def create_environment(self, config, provider):
        return DirectResponseEnvironment(policy=DirectResponsePolicy(),
                                         participants={"assistant": ModelParticipant("assistant", provider)})

    def evaluation_context(self, config):
        from lexverse.config import ConfigError
        attempts = config.evaluation.get("max_attempts", 3)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ConfigError("evaluation.max_attempts must be a positive integer")
        return {"max_attempts": attempts, "timeout_sec": config.evaluation.get("timeout_sec")}

    def create_verifier(self, config=None):
        return PLawBenchTaskVerifier(offline=(config or {}).get("offline", False))

    def create_aggregator(self):
        return PLawBenchAggregator()


PLUGIN = PLawBenchPlugin()
