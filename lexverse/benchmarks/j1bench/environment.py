from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any

from lexverse.benchmarks.j1bench.tasks import J1BenchCase
from lexverse.config import ResolvedConfig
from lexverse.environments.base import ExecutionEnvironment
from lexverse.interaction.schema import EnvironmentResult
from lexverse.runtime.errors import ArtifactMissingError, TrialError
from lexverse.runtime.executor import ExecutionResult, run_subprocess
from lexverse.runtime.sources import UpstreamSource
from lexverse.benchmarks.j1bench import UPSTREAM as J1BENCH
from lexverse.tasks.schema import LexVerseTask

from .scenarios import require_supported
from .tasks import J1BenchLoader, resolve_case_database


@dataclass
class RoleConfig:
    agent_alias: str
    engine_alias: str = "Engine.GPT4o_1120"
    api_key: str | None = None
    api_base: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    generation_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class J1BenchRunConfig:
    scenario: str
    roles: dict[str, RoleConfig]
    max_conversation_turn: int | None = None
    summary: RoleConfig | None = None
    request_timeout_sec: float | None = None

    def __post_init__(self) -> None:
        spec = require_supported(self.scenario)
        missing = set(spec.roles) - set(self.roles)
        extra = set(self.roles) - set(spec.roles)
        if missing or extra:
            raise TrialError(
                f"J1Bench {self.scenario} role mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if self.max_conversation_turn is None:
            self.max_conversation_turn = spec.default_max_turns


@dataclass
class J1BenchRunResult:
    scenario: str
    case_id: str
    save_path: Path
    dialog_history: list[dict[str, Any]]
    execution: ExecutionResult
    raw_record: dict[str, Any] = field(default_factory=dict)


def _role_args(prefix: str, cfg: RoleConfig) -> list[str]:
    # API keys deliberately travel through the subprocess environment, not argv.
    args: list[str] = []
    if cfg.api_base is not None:
        args += [f"--{prefix}_openai_api_base", cfg.api_base]
    if cfg.model_name is not None:
        args += [f"--{prefix}_openai_model_name", cfg.model_name]
    if cfg.temperature is not None:
        args += [f"--{prefix}_temperature", str(cfg.temperature)]
    if cfg.max_tokens is not None:
        args += [f"--{prefix}_max_tokens", str(cfg.max_tokens)]
    args += [f"--{prefix}_lexverse_generation_parameters", json.dumps(cfg.generation_parameters)]
    return args


def _read_dialog_history(save_path: Path,
                         expected_case_id: str) -> tuple[list[dict], dict]:
    if not save_path.exists():
        raise ArtifactMissingError(str(save_path))
    with save_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if str(record.get("case_id")) == str(expected_case_id):
                return list(record.get("dialog_history", [])), record
    raise TrialError(f"case_id {expected_case_id!r} not in {save_path}")


def build_command(upstream_root: Path, case_file: Path, save_path: Path,
                  config: J1BenchRunConfig,
                  python_executable: str = "python") -> list[str]:
    spec = require_supported(config.scenario)
    upstream_root = upstream_root.resolve()
    argv = [
        python_executable, str(upstream_root / "src" / "run.py"),
        "--scenario", spec.alias,
        "--case_database", str(case_file),
        "--save_path", str(save_path),
        "--max_conversation_turn", str(config.max_conversation_turn),
        "--max_workers", "1",
    ]
    for role in spec.roles:
        role_config = config.roles[role]
        argv += [f"--{role}", role_config.agent_alias]
        argv += _role_args(role, role_config)
    return argv


class J1BenchRunner:
    def __init__(self, scenario: str, upstream: UpstreamSource = J1BENCH,
                 offline: bool = False, python_executable: str = "python"):
        self.scenario = require_supported(scenario).name
        self._python = python_executable
        self._cache = upstream.ensure_patched(
            Path(__file__).parent / "patches", offline=offline
        )
        self._src_root = self._cache / "src"
        if not (self._src_root / "run.py").exists():
            raise TrialError(f"upstream missing src/run.py at {self._src_root}")

    async def run(self, case: J1BenchCase, work_dir: Path,
                  config: J1BenchRunConfig, *, timeout_sec: float | None = None,
                  extra_env: dict[str, str] | None = None) -> J1BenchRunResult:
        if case.scenario != self.scenario or config.scenario != self.scenario:
            raise TrialError(
                f"scenario mismatch: runner={self.scenario}, case={case.scenario}, "
                f"config={config.scenario}"
            )
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        case_file = work_dir / "input.json"
        case_file.write_text(json.dumps(case.raw, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        save_path = work_dir / "dialog_history.jsonl"
        # The subprocess cwd is this Trial directory, so benchmark inputs and
        # outputs are deliberately passed as local filenames. This keeps logs
        # and commands portable without reintroducing duplicated relative paths.
        argv = build_command(
            self._cache, Path(case_file.name), Path(save_path.name),
            config, self._python,
        )
        env = dict(os.environ)
        env["J1BENCH_ROOT"] = str(self._cache)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self._src_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        self._inject_credentials(env, config)
        if extra_env:
            env.update(extra_env)
        execution = await run_subprocess(
            command=argv, work_dir=work_dir, timeout_sec=timeout_sec, env=env,
            expected_artifacts={"dialog_history": save_path.name},
        )
        dialog_history, raw = _read_dialog_history(save_path, case.id)
        return J1BenchRunResult(self.scenario, case.id, save_path,
                                dialog_history, execution, raw)

    @staticmethod
    def _inject_credentials(env: dict[str, str], config: J1BenchRunConfig) -> None:
        env.pop("J1BENCH_REQUEST_TIMEOUT", None)
        if config.request_timeout_sec is not None:
            env["J1BENCH_REQUEST_TIMEOUT"] = str(config.request_timeout_sec)
        for name in list(env):
            if name.startswith("J1BENCH_") and name.endswith("_API_KEY"):
                del env[name]
        for role, cfg in config.roles.items():
            if cfg.api_key:
                env[f"J1BENCH_{role.upper()}_API_KEY"] = cfg.api_key
        if config.summary is not None:
            summary = config.summary
            env["J1BENCH_SUMMARY_API_KEY"] = summary.api_key
            env["J1BENCH_SUMMARY_API_BASE"] = summary.api_base or ""
            env["J1BENCH_SUMMARY_MODEL"] = summary.model_name
            env["J1BENCH_SUMMARY_PARAMETERS"] = json.dumps(summary.generation_parameters)


class J1BenchEnvironment(ExecutionEnvironment):
    name = "j1bench_upstream"

    def __init__(self, config: ResolvedConfig) -> None:
        self.config = config
        self.connections = {}
        self._loaders: dict[str, J1BenchLoader] = {}
        self._runners: dict[str, J1BenchRunner] = {}

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
                offline=bool(self.config.generation.get("offline", False)),
                python_executable=sys.executable,
            )
        case = self._loaders[scenario].by_id(task.source.sample_id)
        result = await self._runners[scenario].run(
            case=case,
            work_dir=work_dir,
            config=self._run_config(scenario),
            timeout_sec=self.config.generation.get("timeout_sec"),
        )
        return EnvironmentResult(
            status="completed",
            answer=None,
            dialog_history=result.dialog_history,
            artifacts={"dialog_history": str(result.save_path)},
            final_state={"scenario": scenario, "case_id": result.case_id},
            raw_output={"official_harness": True},
        )

    def _case_database(self, scenario: str) -> Path:
        configured_paths = self.config.benchmark.get("case_databases") or {}
        configured_path = configured_paths.get(scenario)
        if not configured_path and self.config.benchmark.get("scenario") == scenario:
            configured_path = self.config.benchmark.get("case_database")
        return (
            Path(configured_path)
            if configured_path
            else resolve_case_database(
                scenario, offline=bool(self.config.generation.get("offline", False))
            )
        )

    def _run_config(self, scenario: str) -> J1BenchRunConfig:
        scenario_spec = require_supported(scenario)
        roles: dict[str, RoleConfig] = {}
        for role_name in scenario_spec.roles:
            connection = self.connections[role_name]
            parameters = connection.default_config
            roles[role_name] = RoleConfig(
                agent_alias=scenario_spec.default_agents[role_name],
                api_key=connection.api_key,
                api_base=connection.base_url,
                model_name=connection.model,
                temperature=float(parameters["temperature"]) if "temperature" in parameters else None,
                max_tokens=int(parameters["max_tokens"]) if "max_tokens" in parameters else None,
                generation_parameters=parameters,
            )
        summary = None
        if scenario in {"KQ", "LC"}:
            connection = self.connections["summary"]
            summary = RoleConfig(
                agent_alias="summary", api_key=connection.api_key,
                api_base=connection.base_url, model_name=connection.model,
                generation_parameters=connection.default_config,
            )
        return J1BenchRunConfig(
            scenario=scenario,
            roles=roles,
            summary=summary,
            request_timeout_sec=self.config.generation.get("model_request_timeout_sec"),
            max_conversation_turn=int(self.config.generation.get(
                "max_conversation_turn", scenario_spec.default_max_turns
            )),
        )
