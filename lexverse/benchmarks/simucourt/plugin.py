"""Single-candidate SimuCourt generation and independent native scoring."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from lexverse.llms.runtime import scoring_identity
from . import UPSTREAM
from .verifier import SimuCourtAggregator, SimuCourtBenchmarkVerifier
from .tasks import PATCHES, SimuCourtImporter, source_inputs


class SimuCourtPlugin(BenchmarkPlugin):
    name = "simucourt"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def validate_model_config(self, bench, models, modern):
        if models.roles:
            raise ConfigError(
                "roles model overrides are currently supported only by J1Bench, DLawBench, Legal-world and MASER"
            )
        if len(models.verifiers) > 2:
            raise ConfigError("SimuCourt accepts one or two models in models.verifiers")
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")

    def create_importer(self, config=None):
        settings = config or {}
        return SimuCourtImporter(settings, offline=settings.get("offline", False))

    def task_inputs(self, config, task_types):
        paths = source_inputs(config.benchmark, offline=True)
        root = UPSTREAM.ensure_patched(PATCHES, offline=True)
        paths.update({
            "prompt:single_candidate": root / "AgentsCourt/single_candidate_prompt.py",
            "prompt:first_system": root / "AgentsCourt/system_prompt/judge_analysis_record.txt",
            "prompt:second_system": root / "AgentsCourt/system_prompt/judge_analysis_record2.txt",
        })
        return paths

    def resolve_verification_inputs(self, config):
        root = UPSTREAM.ensure_patched(PATCHES, offline=True)
        names = ("judge_result_process", "evaluation_laws", "evaluation_crimial", "evaluation_other")
        return {name: root / "AgentsCourt" / f"{name}.py" for name in names}

    def create_native_runner(self, config):
        from .runner import SimuCourtRunner
        return SimuCourtRunner()

    def verification_models(self, config):
        if len(config.models.verifiers) not in (1, 2):
            raise ConfigError("SimuCourt requires one or two models in models.verifiers")
        self.build_verification_context(config)
        models = {"judge": config.models.verifiers[0]}
        if len(config.models.verifiers) == 2:
            models["other_judge"] = config.models.verifiers[1]
        return models

    def configure_verification_context(self, config, providers):
        judge = providers["judge"]
        other = providers.get("other_judge", judge)
        other_model = config.models.verifiers[-1]
        judge_config = {
            **asdict(config.models.verifiers[0]),
            "endpoint_sha256": scoring_identity(judge)["endpoint_sha256"],
        }
        if len(config.models.verifiers) == 2:
            judge_config["other_judge"] = {
                **asdict(other_model),
                "endpoint_sha256": scoring_identity(other)["endpoint_sha256"],
            }
        judge_model = (
            config.models.verifiers[0].name
            if len(config.models.verifiers) == 1
            else f"{config.models.verifiers[0].name},{other_model.name}"
        )
        return {
            **self.build_verification_context(config),
            "judge_provider": judge,
            "other_judge_provider": other,
            "judge_model": judge_model,
            "judge_config": judge_config,
        }

    def build_verification_context(self, config):
        attempts = config.verification.get("max_attempts", 3)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ConfigError("verification.max_attempts must be a positive integer")
        return {"max_attempts": attempts, "timeout_sec": config.verification.get("timeout_sec")}

    def create_verifier(self, config=None):
        return SimuCourtBenchmarkVerifier(offline=(config or {}).get("offline", False))

    def summary_metadata(self, aggregate):
        return {"stages": aggregate.get("stages", {})}

    def create_aggregator(self):
        return SimuCourtAggregator()


PLUGIN = SimuCourtPlugin()
