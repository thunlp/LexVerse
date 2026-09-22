from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.executor import run_subprocess
from lexverse.runtime.upstream import J1BENCH, UpstreamSource

from .scenarios import require_supported


@dataclass
class J1BenchEvalConfig:
    scoring_api_key: str
    scoring_api_base: str | None = None
    scoring_model: str = "gpt-4o-mini"


@dataclass
class J1BenchEvalResult:
    scenario: str
    model_name: str
    intermediate_dir: Path
    final_json: Path
    per_case: list[dict]
    aggregate: dict


class J1BenchEvaluator:
    def __init__(self, scenario: str, upstream: UpstreamSource = J1BENCH,
                 offline: bool = False, python_executable: str = "python",
                 case_database: Path | None = None):
        self.scenario = require_supported(scenario).name
        self._python = python_executable
        self._cache = upstream.ensure_patched(
            Path(__file__).parent / "patches", offline=offline
        )
        from .runner import _shim_engine_init
        _shim_engine_init(self._cache / "src" / "engine" / "__init__.py")
        self._eval_py = (
            self._cache / "src" / "Eval" / "bench" / self.scenario
            / f"{self.scenario}.py"
        )
        if not self._eval_py.exists():
            raise EvaluationError(
                f"upstream missing evaluator for {self.scenario}: {self._eval_py}"
            )
        self._ensure_ground_truth(case_database)

    def _ensure_ground_truth(self, case_database: Path | None) -> None:
        target = (self._cache / "src" / "data" / "case"
                  / f"J1-Eval_{self.scenario}.jsonl")
        if case_database is None or not Path(case_database).is_file():
            raise EvaluationError(
                f"download J1-Eval_{self.scenario}.jsonl from "
                "https://huggingface.co/datasets/CimoInkPool/J1-Eval_Dataset/tree/main "
                "and pass case_database explicitly"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case_database, target)

    async def evaluate(self, model_name: str, dialog_history_path: Path,
                       output_dir: Path, config: J1BenchEvalConfig, *,
                       timeout_sec: float = 1800.0) -> J1BenchEvalResult:
        output_dir = output_dir.resolve()
        dialog_history_path = dialog_history_path.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        dh_root = output_dir / "dialog_history"
        int_root = output_dir / "intermediate"
        fin_root = output_dir / "final"
        (dh_root / model_name).mkdir(parents=True, exist_ok=True)
        int_root.mkdir(parents=True, exist_ok=True)
        fin_root.mkdir(parents=True, exist_ok=True)
        dest = dh_root / model_name / f"{self.scenario}_dialog_history.jsonl"
        shutil.copyfile(dialog_history_path, dest)

        env = dict(os.environ)
        env["J1BENCH_ROOT"] = str(self._cache)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self._cache / "src"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env["OPENAI_API_KEY"] = config.scoring_api_key
        if config.scoring_api_base:
            env["OPENAI_API_BASE"] = config.scoring_api_base
        env["J1BENCH_SCORING_MODEL"] = config.scoring_model
        await run_subprocess(
            command=[self._python, str(self._eval_py),
                     "--dialog_history_dir", dh_root.name,
                     "--intermediate_eval", int_root.name,
                     "--final_eval", fin_root.name],
            work_dir=output_dir, timeout_sec=timeout_sec, env=env,
        )

        intermediate = int_root / model_name / self.scenario
        per_case = _read_per_case(intermediate)
        # Every upstream evaluator writes <model>_final.json.
        final_json = fin_root / f"{model_name}_final.json"
        if not final_json.is_file():
            raise EvaluationError(
                f"J1Bench {self.scenario} evaluator produced no final result: "
                f"{final_json}"
            )
        aggregate = json.loads(final_json.read_text(encoding="utf-8"))
        return J1BenchEvalResult(self.scenario, model_name, intermediate,
                                 final_json, per_case, aggregate)


def _read_per_case(int_dir: Path) -> list[dict]:
    if not int_dir.exists():
        return []
    out = []
    for path in sorted(int_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value.setdefault("case_id", path.stem)
                out.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    return out
