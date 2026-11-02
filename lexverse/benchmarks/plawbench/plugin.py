from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from . import UPSTREAM
from .verifier import PLawBenchAggregator, PLawBenchVerifier
from .tasks import FILES, PLawBenchImporter


class PLawBenchPlugin(BenchmarkPlugin):
    name = "plawbench"
    supported_agents = frozenset({"static", "vanilla", "external"})
    default_agent = "static"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        return PLawBenchImporter(offline=(config or {}).get("offline", False))

    def task_inputs(self, config, task_types):
        importer = PLawBenchImporter(offline=True)
        return {filename: importer.files[filename] for kind in task_types for filename in FILES[kind].values()}


    def build_verification_context(self, config):
        from lexverse.config import ConfigError
        if len(config.models.verifiers) != 1:
            raise ConfigError("PLawBench requires one model in models.verifiers")
        attempts = config.verification.get("max_attempts", 3)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ConfigError("verification.max_attempts must be a positive integer")
        return {"max_attempts": attempts, "timeout_sec": config.verification.get("timeout_sec")}

    def create_verifier(self, config=None):
        return PLawBenchVerifier(offline=(config or {}).get("offline", False))

    def create_aggregator(self):
        return PLawBenchAggregator()


PLUGIN = PLawBenchPlugin()
