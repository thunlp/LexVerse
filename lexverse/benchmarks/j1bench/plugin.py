from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.benchmarks.j1bench import UPSTREAM as J1BENCH
from lexverse.tasks.bundle import TaskSelection

from .evaluation import J1BenchAggregator, J1BenchTaskVerifier
from .scenarios import ALL_SCENARIO_NAMES, require_supported
from .tasks import J1BenchImporter, resolve_case_database


class J1BenchPlugin(BenchmarkPlugin):
    name = "j1bench"
    upstream = J1BENCH
    source_version = J1BENCH.commit
    default_policy = "j1bench_ci"
    default_environment = "j1bench_upstream"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_bundle(self, config):
        # Freeze the resolved gated dataset locations, including default cache paths.
        config.benchmark["case_databases"] = {
            scenario: str(path.resolve()) for scenario, path in self.evaluation_inputs(config).items()
        }
        return super().create_bundle(config)

    def create_importer(self, config=None):
        config = config or {}
        legacy_scenario = config.get("scenario")
        scenarios = config.get("tasks") or config.get("scenarios") or (
            [legacy_scenario] if legacy_scenario else list(ALL_SCENARIO_NAMES)
        )
        if not isinstance(scenarios, list) or not scenarios:
            from lexverse.config import ConfigError
            raise ConfigError("j1bench benchmark.scenarios must be a non-empty list")
        for scenario in scenarios:
            try:
                require_supported(scenario)
            except (KeyError, NotImplementedError) as exc:
                from lexverse.config import ConfigError
                raise ConfigError(str(exc)) from exc

        configured_paths = config.get("case_databases") or {}
        if not isinstance(configured_paths, dict):
            from lexverse.config import ConfigError
            raise ConfigError("j1bench benchmark.case_databases must be a mapping")
        if legacy_scenario and config.get("case_database"):
            configured_paths = {
                **configured_paths, legacy_scenario: config["case_database"]
            }
        case_databases = {
            scenario: (
                Path(configured_paths[scenario])
                if configured_paths.get(scenario)
                else resolve_case_database(
                    scenario, offline=bool(config.get("offline", False))
                )
            )
            for scenario in scenarios
        }
        return J1BenchImporter(
            scenarios=scenarios,
            case_databases=case_databases,
            catalog=self.load_catalog(),
        )

    def task_inputs(self, config, task_types):
        return {kind: Path(config.benchmark["case_databases"][kind]) for kind in task_types}

    def create_environment(self, config, provider):
        del provider
        from .environment import J1BenchEnvironment
        return J1BenchEnvironment(config)

    def evaluation_context(self, config):
        return {
            "case_database": config.benchmark.get("case_database"),
            "case_databases": config.benchmark.get("case_databases"),
            "model_name": config.benchmark.get("model_tag", config.models.default.name),
            "offline": bool(config.generation.get("offline", False)),
            "timeout_sec": config.evaluation.get("timeout_sec"),
        }

    def evaluation_inputs(self, config):
        importer = self.create_importer({
            **config.benchmark, "offline": bool(config.generation.get("offline", False)),
            "tasks": TaskSelection.from_config(config.benchmark, config.generation).task_types,
        })
        return {scenario: Path(path) for scenario, path in importer.case_databases.items()}

    def create_verifier(self, config=None):
        return J1BenchTaskVerifier()

    def create_aggregator(self):
        return J1BenchAggregator()

PLUGIN = J1BenchPlugin()
