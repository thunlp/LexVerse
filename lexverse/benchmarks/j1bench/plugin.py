from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.runtime.upstream import J1BENCH

from .importer import J1BenchImporter
from .verifier import J1BenchTaskVerifier
from .aggregate import J1BenchAggregator
from .dataset import resolve_case_database
from .scenarios import ALL_SCENARIO_NAMES, require_supported


class J1BenchPlugin(BenchmarkPlugin):
    name = "j1bench"
    source_version = J1BENCH.commit
    default_policy = "j1bench_ci"
    default_environment = "j1bench_upstream"
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_importer(self, config=None):
        config = config or {}
        legacy_scenario = config.get("scenario")
        scenarios = config.get("scenarios") or (
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

    def create_verifier(self, config=None):
        return J1BenchTaskVerifier()

    def create_environment(self, config, provider):
        del provider
        from .environment import J1BenchEnvironment
        return J1BenchEnvironment(config)

    def create_aggregator(self):
        return J1BenchAggregator()

PLUGIN = J1BenchPlugin()
