"""MILE benchmark adapter for native generation and evaluation."""
from __future__ import annotations

from pathlib import Path
import sys

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from lexverse.tasks.bundle import TaskSelection
from . import UPSTREAM
from .environment import MILEEnvironment
from .evaluation import MILEAggregator, MILETaskVerifier
from .tasks import MILEImporter, TASK_TYPE, max_turns, source_inputs


class MILEPlugin(BenchmarkPlugin):
    name = "maser"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    default_policy = "mile_native"
    default_environment = "mile_native"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def validate_model_config(self, bench, models, modern):
        if set(models.roles) - {"lawyer", "client", "supervisor"}:
            raise ConfigError("MILE roles must be lawyer, client or supervisor")
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")
        if not {"client", "supervisor"} <= set(models.roles):
            raise ConfigError("MILE requires explicit models.roles.client and supervisor")
        if len(models.evaluators) != 1:
            raise ConfigError("MILE requires one model in models.evaluators")

    def generation_models(self, config, bundle):
        return {
            role: config.models.roles.get(role, config.models.default)
            for role in ("lawyer", "client", "supervisor")
        }

    def configure_environment(self, config, providers):
        environment = self.create_environment(config, None)
        environment.connections = providers
        return environment

    def tested_model(self, config):
        return config.models.roles.get("lawyer", config.models.default)

    def evaluation_models(self, config):
        if len(config.models.evaluators) != 1:
            raise ConfigError("MILE requires one model in models.evaluators")
        self.evaluation_context(config)
        return {"judge": config.models.evaluators[0]}

    def configure_evaluation_context(self, config, providers):
        scoring = providers["judge"]
        return {
            **self.evaluation_context(config),
            "scoring_api_key": scoring.api_key,
            "scoring_api_base": scoring.base_url,
            "scoring_model": scoring.model,
            "scoring_parameters": {"temperature": 0, **scoring.default_config},
            "request_timeout_sec": config.evaluation.get("model_request_timeout_sec"),
            "python_executable": sys.executable,
        }

    def create_importer(self, config=None):
        return MILEImporter(config)

    def create_bundle(self, config):
        self.validate_model_config(config.benchmark, config.models, True)
        self.evaluation_context(config)
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

    def evaluation_inputs(self, config):
        return self.task_inputs(config, [TASK_TYPE])

    def create_environment(self, config, provider):
        return MILEEnvironment(config)

    def evaluation_context(self, config):
        attempts = config.evaluation.get("max_attempts", 3)
        if type(attempts) is not int or attempts < 1:
            raise ConfigError("evaluation.max_attempts must be a positive integer")
        return {
            "offline": bool(config.generation.get("offline", False)),
            "max_attempts": attempts,
            "timeout_sec": config.evaluation.get("timeout_sec"),
            "concurrency": config.evaluation.get("concurrency", 4),
        }

    def create_verifier(self, config=None):
        return MILETaskVerifier()

    def create_aggregator(self):
        return MILEAggregator()


MASERPlugin = MILEPlugin
PLUGIN = MILEPlugin()
