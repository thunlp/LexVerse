from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from . import UPSTREAM
from .environment import DLawBenchEnvironment
from .evaluation import DLawBenchAggregator, DLawBenchTaskVerifier
from .tasks import DLawBenchImporter, source_inputs


class DLawBenchPlugin(BenchmarkPlugin):
    name = "dlawbench"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    default_policy = "dlawbench_native"
    default_environment = "dlawbench_native"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def validate_model_config(self, bench, models, modern):
        if set(models.roles) - {"lawyer", "client"}:
            raise ConfigError("DLawBench roles must be lawyer or client")
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")

    def generation_models(self, config, bundle):
        return {role: config.models.roles.get(role, config.models.default)
                for role in ("lawyer", "client")}

    def configure_environment(self, config, providers):
        environment = self.create_environment(config, None)
        environment.connections = providers
        return environment

    def tested_model(self, config):
        return config.models.roles.get("lawyer", config.models.default)

    def evaluation_models(self, config):
        return self.evaluation_context(config)["judge_models"]

    def configure_evaluation_context(self, config, providers):
        context = self.evaluation_context(config)
        context.pop("judge_models")
        return {**context, "judge_connections": providers,
                "timeout_sec": config.evaluation.get("timeout_sec")}

    def create_importer(self, config=None):
        return DLawBenchImporter(offline=(config or {}).get("offline", False))

    def create_bundle(self, config):
        self.evaluation_context(config)
        turns = config.generation.get("max_turns", 10)
        if type(turns) is not int or turns < 1:
            raise ConfigError("generation.max_turns must be a positive integer")
        return super().create_bundle(config)

    def task_inputs(self, config, task_types):
        return source_inputs(UPSTREAM.ensure(offline=True))

    def create_environment(self, config, provider):
        return DLawBenchEnvironment(config)

    def evaluation_context(self, config):
        mode = config.evaluation.get("mode")
        if mode not in {"single", "panel"}:
            raise ConfigError("DLawBench requires explicit evaluation.mode: single or panel")
        attempts = config.evaluation.get("max_attempts", 3)
        if type(attempts) is not int or attempts < 1:
            raise ConfigError("evaluation.max_attempts must be a positive integer")
        if mode == "single":
            if len(config.models.evaluators) != 1:
                raise ConfigError("DLawBench single mode requires one model in models.evaluators")
            models = {"judge": config.models.evaluators[0]}
        else:
            judges = config.models.evaluators
            if len(config.models.evaluators) != 3:
                raise ConfigError("native panel requires three complete judge model mappings")
            models = {}
            for model in judges:
                if model.name in models:
                    raise ConfigError("panel judge names must be unique")
                models[model.name] = model
        return {"mode": mode, "max_attempts": attempts, "judge_models": models}

    def create_verifier(self, config=None):
        return DLawBenchTaskVerifier(offline=(config or {}).get("offline", False))

    def create_aggregator(self):
        return DLawBenchAggregator()


PLUGIN = DLawBenchPlugin()
