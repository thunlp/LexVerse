from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from . import UPSTREAM
from .environment import DLawBenchEnvironment
from .verifier import DLawBenchAggregator, DLawBenchVerifier
from .tasks import DLawBenchImporter, source_inputs


class DLawBenchPlugin(BenchmarkPlugin):
    name = "dlawbench"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        return DLawBenchImporter(offline=(config or {}).get("offline", False))

    def create_bundle(self, config):
        turns = config.generation.get("max_turns", 10)
        if type(turns) is not int or turns < 1:
            raise ConfigError("generation.max_turns must be a positive integer")
        return super().create_bundle(config)

    def task_inputs(self, config, task_types):
        return source_inputs(UPSTREAM.ensure(offline=True))

    def create_native_runner(self, config):
        return DLawBenchEnvironment(config)

    def build_verification_context(self, config):
        mode = config.verification.get("mode")
        if mode not in {"single", "panel"}:
            raise ConfigError("DLawBench requires explicit verification.mode: single or panel")
        attempts = config.verification.get("max_attempts", 3)
        if type(attempts) is not int or attempts < 1:
            raise ConfigError("verification.max_attempts must be a positive integer")
        if mode == "single":
            if len(config.models.verifiers) != 1:
                raise ConfigError("DLawBench single mode requires one model in models.verifiers")
            models = {"judge": config.models.verifiers[0]}
        else:
            judges = config.models.verifiers
            if len(config.models.verifiers) != 3:
                raise ConfigError("native panel requires three complete judge model mappings")
            models = {}
            for model in judges:
                if model.name in models:
                    raise ConfigError("panel judge names must be unique")
                models[model.name] = model
        return {"mode": mode, "max_attempts": attempts, "judge_models": models}

    def create_verifier(self, config=None):
        return DLawBenchVerifier(offline=(config or {}).get("offline", False))

    def create_aggregator(self):
        return DLawBenchAggregator()

    def validate_model_config(self, bench, models, modern):
        if set(models.roles) - {"lawyer", "client"}:
            raise ConfigError("DLawBench roles must be lawyer or client")
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")

    def generation_models(self, config, bundle):
        return {role: config.models.roles.get(role, config.models.default)
                for role in ("lawyer", "client")}

    def tested_model(self, config):
        return config.models.roles.get("lawyer", config.models.default)

PLUGIN = DLawBenchPlugin()
