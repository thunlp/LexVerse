from __future__ import annotations

from pathlib import Path

from lexverse.benchmarks.plugin import BenchmarkPlugin
from lexverse.benchmarks.j1bench import UPSTREAM as J1BENCH
from lexverse.tasks.bundle import TaskSelection

from .verifier import J1BenchAggregator, J1BenchVerifier
from .scenarios import ALL_SCENARIO_NAMES, require_supported
from .tasks import J1BenchImporter, resolve_case_database


class J1BenchPlugin(BenchmarkPlugin):
    name = "j1bench"
    upstream = J1BENCH
    source_version = J1BENCH.commit
    catalog_path = Path(__file__).with_name("catalog.yaml")

    def create_bundle(self, config):
        # Freeze the resolved gated dataset locations, including default cache paths.
        config.benchmark["case_databases"] = {
            scenario: str(path.resolve()) for scenario, path in self.resolve_verification_inputs(config).items()
        }
        return super().create_bundle(config)

    def create_importer(self, config=None):
        config = config or {}
        if set(config) & {"scenario", "scenarios", "case_database"}:
            from lexverse.config import ConfigError
            raise ConfigError("Use benchmark.tasks and benchmark.case_databases for J1Bench")
        scenarios = config.get("tasks") or list(ALL_SCENARIO_NAMES)
        if not isinstance(scenarios, list) or not scenarios:
            from lexverse.config import ConfigError
            raise ConfigError("j1bench benchmark.tasks must be a non-empty list")
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

    def create_native_runner(self, config):
        from .environment import J1BenchEnvironment
        return J1BenchEnvironment(config)

    def build_verification_context(self, config):
        from lexverse.config import ConfigError
        if len(config.models.verifiers) != 1:
            raise ConfigError("J1Bench requires one model in models.verifiers")
        return {
            "case_databases": config.benchmark.get("case_databases"),
            "model_name": config.benchmark.get("model_tag", config.models.default.name),
            "offline": bool(config.generation.get("offline", False)),
            "timeout_sec": config.verification.get("timeout_sec"),
        }

    def resolve_verification_inputs(self, config):
        importer = self.create_importer({
            **config.benchmark, "offline": bool(config.generation.get("offline", False)),
            "tasks": TaskSelection.from_config(config.benchmark, config.generation).task_types,
        })
        return {scenario: Path(path) for scenario, path in importer.case_databases.items()}

    def create_verifier(self, config=None):
        return J1BenchVerifier()

    def create_aggregator(self):
        return J1BenchAggregator()

    def validate_model_config(self, bench, models, modern):
        from lexverse.config import ConfigError
        from .scenarios import SCENARIOS
        if len(models.verifiers) > 1:
            raise ConfigError("this benchmark accepts one model in models.verifiers")
        if models.roles:
            names = TaskSelection.from_config(bench, {}).select_types(list(SCENARIOS))
            supported_roles = {role for name in names for role in SCENARIOS[name].roles}
            if set(models.roles) - supported_roles:
                raise ConfigError(f"roles not present in the selected scenarios: {sorted(set(models.roles) - supported_roles)}")

    def generation_models(self, config, bundle):
        role_names = dict.fromkeys(role for task in bundle.tasks
                                   for role in require_supported(task.source.task_type).roles)
        models = {role: config.models.roles.get(role, config.models.default) for role in role_names}
        if any(task.source.task_type in {"KQ", "LC"} for task in bundle.tasks):
            models["summary"] = config.models.summary
        return models

PLUGIN = J1BenchPlugin()
