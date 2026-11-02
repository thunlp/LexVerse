"""MILE benchmark adapter for native generation and evaluation."""
from __future__ import annotations

from pathlib import Path
import sys

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from lexverse.tasks.bundle import TaskSelection
from . import UPSTREAM
from .environment import MILEEnvironment
from .verifier import MILEAggregator, MILEBenchmarkVerifier
from .tasks import MILEImporter, TASK_TYPE, max_turns, source_inputs


class MILEPlugin(BenchmarkPlugin):
    name = "maser"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def validate_model_config(self, bench, models, modern):
        if set(models.roles) - {"lawyer", "client", "supervisor"}:
            raise ConfigError("MILE roles must be lawyer, client or supervisor")
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")
        if not {"client", "supervisor"} <= set(models.roles):
            raise ConfigError("MILE requires explicit models.roles.client and supervisor")
        if len(models.verifiers) != 1:
            raise ConfigError("MILE requires one model in models.verifiers")

    def generation_models(self, config, bundle):
        return {
            role: config.models.roles.get(role, config.models.default)
            for role in ("lawyer", "client", "supervisor")
        }

    def tested_model(self, config):
        return config.models.roles.get("lawyer", config.models.default)

    def verification_models(self, config):
        if len(config.models.verifiers) != 1:
            raise ConfigError("MILE requires one model in models.verifiers")
        self.build_verification_context(config)
        return {"judge": config.models.verifiers[0]}

    def configure_verification_context(self, config, providers):
        scoring = providers["judge"]
        return {
            **self.build_verification_context(config),
            "scoring_api_key": scoring.api_key,
            "scoring_api_base": scoring.base_url,
            "scoring_model": scoring.model,
            "scoring_parameters": {"temperature": 0, **scoring.default_config},
            "request_timeout_sec": config.verification.get("model_request_timeout_sec"),
            "python_executable": sys.executable,
        }

    def create_importer(self, config=None):
        return MILEImporter(config)

    def create_bundle(self, config):
        self.validate_model_config(config.benchmark, config.models, True)
        self.build_verification_context(config)
        config.benchmark["max_turns"] = max_turns(config.generation)
        return super().create_bundle(config)

    def restore_bundle(self, config, references):
        config.benchmark["max_turns"] = max_turns(config.generation)
        return super().restore_bundle(config, references)

    def task_inputs(self, config, task_types):
        return source_inputs({
            **config.benchmark,
            "offline": bool(config.generation.get("offline", False)),
        })

    def resolve_verification_inputs(self, config):
        return self.task_inputs(config, [TASK_TYPE])

    def create_native_runner(self, config):
        return MILEEnvironment(config)

    def build_verification_context(self, config):
        attempts = config.verification.get("max_attempts", 3)
        if type(attempts) is not int or attempts < 1:
            raise ConfigError("verification.max_attempts must be a positive integer")
        return {
            "offline": bool(config.generation.get("offline", False)),
            "max_attempts": attempts,
            "timeout_sec": config.verification.get("timeout_sec"),
            "concurrency": config.verification.get("concurrency", 4),
        }

    def create_verifier(self, config=None):
        return MILEBenchmarkVerifier()

    def create_aggregator(self):
        return MILEAggregator()


MASERPlugin = MILEPlugin
PLUGIN = MILEPlugin()
