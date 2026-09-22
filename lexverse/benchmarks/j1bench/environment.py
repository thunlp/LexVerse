from __future__ import annotations

import sys
from pathlib import Path

from lexverse.config import ResolvedConfig, load_profile, parse_model_config
from lexverse.environments.base import ExecutionEnvironment
from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask

from .loader import J1BenchLoader
from .dataset import resolve_case_database
from .runner import J1BenchRunConfig, J1BenchRunner, RoleConfig
from .scenarios import require_supported


class J1BenchEnvironment(ExecutionEnvironment):
    name = "j1bench_upstream"

    def __init__(self, config: ResolvedConfig) -> None:
        self.config = config
        self._loaders: dict[str, J1BenchLoader] = {}
        self._runners: dict[str, J1BenchRunner] = {}

    def _case_database(self, scenario: str) -> Path:
        configured_paths = self.config.benchmark.get("case_databases") or {}
        configured_path = configured_paths.get(scenario)
        if not configured_path and self.config.benchmark.get("scenario") == scenario:
            configured_path = self.config.benchmark.get("case_database")
        return (
            Path(configured_path)
            if configured_path
            else resolve_case_database(
                scenario, offline=bool(self.config.execution.get("offline", False))
            )
        )

    def _run_config(self, scenario: str) -> J1BenchRunConfig:
        scenario_spec = require_supported(scenario)
        primary = load_profile(self.config.model.profile)
        configured_roles = self.config.raw.get("roles") or {}
        roles: dict[str, RoleConfig] = {}
        for role_name in scenario_spec.roles:
            raw = configured_roles.get(role_name) or {}
            model = parse_model_config(raw.get("model", {
                "profile": self.config.model.profile,
                "name": self.config.model.name,
                "parameters": self.config.model.parameters,
            }), field=f"roles.{role_name}.model")
            connection = (
                primary if model.profile == self.config.model.profile
                else load_profile(model.profile)
            )
            roles[role_name] = RoleConfig(
                agent_alias=scenario_spec.default_agents[role_name],
                api_key=connection.api_key,
                api_base=connection.base_url,
                model_name=model.name,
                max_tokens=int(model.parameters.get(
                    "max_tokens", raw.get("max_tokens", 512)
                )),
            )
        return J1BenchRunConfig(
            scenario=scenario,
            roles=roles,
            max_conversation_turn=int(self.config.execution.get(
                "max_conversation_turn", scenario_spec.default_max_turns
            )),
        )

    async def run(self, task: LexVerseTask, work_dir: Path) -> EnvironmentResult:
        scenario = task.source.task_type
        require_supported(scenario)
        if scenario not in self._loaders:
            self._loaders[scenario] = J1BenchLoader(
                self._case_database(scenario), scenario=scenario
            )
        if scenario not in self._runners:
            self._runners[scenario] = J1BenchRunner(
                scenario,
                offline=bool(self.config.execution.get("offline", False)),
                python_executable=sys.executable,
            )
        case = self._loaders[scenario].by_id(task.source.sample_id)
        result = await self._runners[scenario].run(
            case=case,
            work_dir=work_dir,
            config=self._run_config(scenario),
            timeout_sec=float(self.config.execution.get("timeout_sec", 600.0)),
        )
        return EnvironmentResult(
            status="completed",
            answer=None,
            dialog_history=result.dialog_history,
            artifacts={"dialog_history": str(result.save_path)},
            final_state={"scenario": scenario, "case_id": result.case_id},
            raw_output={"official_harness": True},
        )
