"""Single-candidate SimuCourt generation and independent native scoring."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from lexverse.environments import DirectResponseEnvironment
from lexverse.interaction.participants import ModelParticipant
from lexverse.interaction.policies import DirectResponsePolicy
from lexverse.providers.runtime import scoring_identity
from . import UPSTREAM
from .evaluation import SimuCourtAggregator, SimuCourtTaskVerifier
from .tasks import PATCHES, SimuCourtImporter, source_inputs


class SimuCourtPlugin(BenchmarkPlugin):
    name = "simucourt"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    default_policy = "direct_response"
    default_environment = "direct_response"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def validate_model_config(self, bench, models, modern):
        if models.roles:
            raise ConfigError(
                "roles model overrides are currently supported only by J1Bench, DLawBench, Legal-world and MASER"
            )
        if len(models.evaluators) > 2:
            raise ConfigError("SimuCourt accepts one or two models in models.evaluators")
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

    def evaluation_inputs(self, config):
        root = UPSTREAM.ensure_patched(PATCHES, offline=True)
        names = ("judge_result_process", "evaluation_laws", "evaluation_crimial", "evaluation_other")
        return {name: root / "AgentsCourt" / f"{name}.py" for name in names}

    def create_environment(self, config, provider):
        return DirectResponseEnvironment(
            policy=DirectResponsePolicy(),
            participants={"assistant": ModelParticipant("assistant", provider)},
        )

    def evaluation_models(self, config):
        if len(config.models.evaluators) not in (1, 2):
            raise ConfigError("SimuCourt requires one or two models in models.evaluators")
        self.evaluation_context(config)
        models = {"judge": config.models.evaluators[0]}
        if len(config.models.evaluators) == 2:
            models["other_judge"] = config.models.evaluators[1]
        return models

    def configure_evaluation_context(self, config, providers):
        judge = providers["judge"]
        other = providers.get("other_judge", judge)
        other_model = config.models.evaluators[-1]
        judge_config = {
            **asdict(config.models.evaluators[0]),
            "endpoint_sha256": scoring_identity(judge)["endpoint_sha256"],
        }
        if len(config.models.evaluators) == 2:
            judge_config["other_judge"] = {
                **asdict(other_model),
                "endpoint_sha256": scoring_identity(other)["endpoint_sha256"],
            }
        judge_model = (
            config.models.evaluators[0].name
            if len(config.models.evaluators) == 1
            else f"{config.models.evaluators[0].name},{other_model.name}"
        )
        return {
            **self.evaluation_context(config),
            "judge_provider": judge,
            "other_judge_provider": other,
            "judge_model": judge_model,
            "judge_config": judge_config,
        }

    def evaluation_context(self, config):
        attempts = config.evaluation.get("max_attempts", 3)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ConfigError("evaluation.max_attempts must be a positive integer")
        return {"max_attempts": attempts, "timeout_sec": config.evaluation.get("timeout_sec")}

    def create_verifier(self, config=None):
        return SimuCourtTaskVerifier(offline=(config or {}).get("offline", False))

    def summary_metadata(self, aggregate):
        return {"stages": aggregate.get("stages", {})}

    def create_aggregator(self):
        return SimuCourtAggregator()


PLUGIN = SimuCourtPlugin()
