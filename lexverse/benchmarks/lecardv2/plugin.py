"""LeCaRDv2 benchmark plugin supporting Mode A (encoder/submission) and Mode B (LLM query rewrite)."""
from __future__ import annotations

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.config import ConfigError
from . import CATALOG_PATH, UPSTREAM
from .environment import LeCaRDv2Environment
from .evaluation import LeCaRDv2Aggregator, LeCaRDv2TaskVerifier
from .tasks import (
    LeCaRDv2Importer, evaluation_source_inputs, source_inputs, validate_benchmark,
)


class LeCaRDv2Plugin(BenchmarkPlugin):
    name = "lecardv2"
    upstream = UPSTREAM
    source_version = UPSTREAM.commit
    default_policy = "lecardv2_ranked_submission"
    default_environment = "lecardv2_retrieval"
    catalog_path = CATALOG_PATH

    def validate_model_config(self, bench, models, modern):
        try:
            validate_benchmark(bench)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if models.roles:
            raise ConfigError(
                "roles model overrides are currently supported only by J1Bench, DLawBench, Legal-world and MASER"
            )
        if models.summary is not None:
            raise ConfigError("models.summary is supported only by J1Bench KQ/LC")
        if modern and models.evaluators:
            raise ConfigError("this benchmark uses its native scorer without models.evaluators")

    def generation_models(self, config, bundle):
        mode = config.benchmark.get("mode", "ranked_submission")
        if mode in {"ranked_submission", "bm25_retrieval", "encoder_retrieval", "keller_subfact_retrieval"}:
            return {}
        return {"default": config.models.default}

    def configure_environment(self, config, providers):
        return self.create_environment(config, providers.get("default"))

    def create_importer(self, config=None):
        settings = dict(config or {})
        return LeCaRDv2Importer(settings, offline=bool(settings.get("offline", False)))

    def task_inputs(self, config, task_types):
        del task_types
        return source_inputs(config.benchmark, offline=True)

    def evaluation_inputs(self, config):
        return evaluation_source_inputs(
            config.benchmark, offline=bool(config.generation.get("offline", False))
        )

    def create_environment(self, config, provider):
        return LeCaRDv2Environment(config, provider)

    def create_verifier(self, config=None):
        return LeCaRDv2TaskVerifier(config=config)

    def create_aggregator(self):
        return LeCaRDv2Aggregator()

    def summary_metadata(self, aggregate):
        return {
            key: aggregate[key]
            for key in (
                "protocol",
                "primary_aggregation",
                "reproduction_mode",
                "candidate_scope",
                "prompt_template",
                "aggregations",
                "zero_gold_query_ids",
                "nonempty_gold_queries",
                "limitations",
            )
            if key in aggregate
        }


PLUGIN = LeCaRDv2Plugin()
