"""Run J1Bench through the pinned upstream harness."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lexverse.benchmarks.j1bench.loader import J1BenchCase
from lexverse.runtime.errors import ArtifactMissingError, TrialError
from lexverse.runtime.executor import ExecutionResult, run_subprocess
from lexverse.runtime.upstream import J1BENCH, UpstreamSource

from .scenarios import require_supported


@dataclass
class RoleConfig:
    agent_alias: str
    engine_alias: str = "Engine.GPT4o_1120"
    api_key: str | None = None
    api_base: str | None = None
    model_name: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class J1BenchRunConfig:
    scenario: str
    roles: dict[str, RoleConfig]
    max_conversation_turn: int | None = None

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


def _role_args(prefix: str, cfg: RoleConfig) -> list[str]:
    # API keys deliberately travel through the subprocess environment, not argv.
    args: list[str] = []
    if cfg.api_base is not None:
        args += [f"--{prefix}_openai_api_base", cfg.api_base]
    if cfg.model_name is not None:
        args += [f"--{prefix}_openai_model_name", cfg.model_name]
    args += [f"--{prefix}_temperature", str(cfg.temperature),
             f"--{prefix}_max_tokens", str(cfg.max_tokens)]
    return args


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
        _shim_engine_init(self._src_root / "engine" / "__init__.py")

    async def run(self, case: J1BenchCase, work_dir: Path,
                  config: J1BenchRunConfig, *, timeout_sec: float = 600.0,
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
        keys = {cfg.api_key for cfg in config.roles.values() if cfg.api_key}
        if len(keys) > 1:
            raise TrialError(
                "J1Bench requires one shared API key across roles because passing "
                "per-role keys in argv would expose secrets"
            )
        if keys:
            env["OPENAI_API_KEY"] = str(keys.pop())
        bases = {cfg.api_base for cfg in config.roles.values() if cfg.api_base}
        if len(bases) == 1:
            env.setdefault("OPENAI_API_BASE", str(bases.pop()))


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


_ENGINE_SHIM_MARKER = "# LEXVERSE-SHIM-V1\n"
_ENGINE_SHIM = _ENGINE_SHIM_MARKER + '''"""Load API engine and optional local engines."""
from .gpt_4o_1120 import GPT_1120Engine
__all__ = ["GPT_1120Engine"]
for _mod, _cls in [
    (".lawllm", "LawLLMEngine"), (".deepseek_v3", "DeepseekEngine"),
    (".ministral8B", "Ministral8BEngine"), (".GLM_4_9B", "GLM9BEngine"),
    (".chatlaw2", "Chatlaw2Engine"), (".qwen3_14B", "qwen3_14BEngine"),
    (".qwen3_32B", "qwen3_32BEngine"), (".gemma12b", "Gemma12BEngine"),
    (".internlm3", "InternLM3Engine"), (".llama33_70B", "LLaMa3_3Engine"),
]:
    try:
        _m = __import__(f"engine{_mod}", fromlist=[_cls])
        globals()[_cls] = getattr(_m, _cls)
        __all__.append(_cls)
    except ImportError:
        pass
'''


def _shim_engine_init(init_path: Path) -> None:
    if init_path.exists():
        try:
            if init_path.read_text(encoding="utf-8").startswith(_ENGINE_SHIM_MARKER):
                return
        except OSError:
            pass
    init_path.write_text(_ENGINE_SHIM, encoding="utf-8")
