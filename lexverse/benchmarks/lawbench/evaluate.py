"""Call LawBench's deterministic per-task scoring functions.

A scoped ``MetaPathFinder`` ensures the evaluator resolves its own top-level
``utils`` package even when another benchmark imported a package of that name.
Prediction files retain the upstream ``origin_prompt/prediction/refr`` shape.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from lexverse.compatibility import rouge_warning
from lexverse.runtime.errors import EvaluationError
from lexverse.runtime.upstream import LAWBENCH, UpstreamSource


# Upstream `evaluation/main.py::funct_dict` — task_id -> (module, function).
FUNCT_TABLE: dict[str, tuple[str, str]] = {
    "1-1": ("ftcs", "compute_ftcs"),
    "1-2": ("jec_kd", "compute_jec_kd"),
    "2-1": ("wsjd", "compute_wsjd"),
    "2-2": ("jdzy", "compute_jdzy"),
    "2-3": ("wbfl", "compute_wbfl"),
    "2-4": ("zxfl", "compute_zxfl"),
    "2-5": ("ydlj", "compute_ydlj"),
    "2-6": ("xxcq", "compute_xxcq"),
    "2-7": ("yqzy", "compute_yqzy"),
    "2-8": ("lblj", "compute_lblj"),
    "2-9": ("sjjc", "compute_sjjc"),
    "2-10": ("sjjc", "compute_cfcy"),
    "3-1": ("ljp_article", "compute_ljp_article"),
    "3-2": ("cjft", "compute_cjft"),
    "3-3": ("ljp_accusation", "compute_ljp_accusation"),
    "3-4": ("ljp_imprison", "compute_ljp_imprison"),
    "3-5": ("ljp_imprison", "compute_ljp_imprison"),
    "3-6": ("jec_ac", "compute_jec_ac"),
    "3-7": ("jetq", "compute_jetq"),
    "3-8": ("flzx", "compute_flzx"),
}


@dataclass(frozen=True)
class LawBenchScore:
    task_id: str
    score: float
    abstention_rate: float


def prediction_to_disk(
    path: Path,
    origin_prompts: list[str],
    predictions: list[str],
    refrs: list[str],
) -> Path:
    if not (len(origin_prompts) == len(predictions) == len(refrs)):
        raise EvaluationError(
            f"origin_prompts/predictions/refrs length mismatch: "
            f"{len(origin_prompts)}/{len(predictions)}/{len(refrs)}"
        )
    data: dict[str, dict] = {}
    for i, (op, pred, refr) in enumerate(zip(origin_prompts, predictions, refrs)):
        data[str(i)] = {
            "origin_prompt": [{"role": "HUMAN", "prompt": op}],
            "prediction": pred,
            "refr": refr,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class LawBenchEvaluator:
    def __init__(self, upstream: UpstreamSource = LAWBENCH, offline: bool = False):
        self._upstream = upstream
        _warn = rouge_warning()
        if _warn:
            print(f"[lexverse] {_warn}", file=sys.stderr)
        self._cache = upstream.ensure_patched(
            Path(__file__).parent / "patches", offline=offline
        )
        self._eval_dir = self._cache / "evaluation"
        if not self._eval_dir.exists():
            raise EvaluationError(f"upstream missing evaluation/ at {self._eval_dir}")
        self._fn_dir = self._eval_dir / "evaluation_functions"
        self._utils_dir = self._eval_dir / "utils"
        if not (self._utils_dir / "function_utils.py").exists():
            raise EvaluationError(
                f"upstream missing utils/function_utils.py at {self._utils_dir}"
            )
        self._fn_cache: dict[str, object] = {}

    # -- public API --------------------------------------------------------

    def _get_function(self, task_id: str):
        if task_id in self._fn_cache:
            return self._fn_cache[task_id]
        if task_id not in FUNCT_TABLE:
            raise EvaluationError(
                f"unknown LawBench task {task_id!r}; "
                f"available: {sorted(FUNCT_TABLE)}"
            )
        mod_name, fn_name = FUNCT_TABLE[task_id]
        path = self._fn_dir / f"{mod_name}.py"
        if not path.exists():
            raise EvaluationError(f"upstream evaluator module not found: {path}")
        try:
            mod = self._import_isolated(mod_name, path)
        except ImportError as exc:
            raise EvaluationError(
                f"upstream evaluator {mod_name} import failed "
                f"(likely missing optional dep for task {task_id}): {exc}"
            ) from exc
        fn = getattr(mod, fn_name)
        self._fn_cache[task_id] = fn
        return fn

    def _import_isolated(self, mod_name: str, path: Path):
        """Import under a private name with a finder resolving `utils` to ours."""
        private = f"_lexverse_lawbench_{mod_name}"
        if private in sys.modules:
            return sys.modules[private]

        finder = _DirFirstFinder(("utils",), self._utils_dir)
        sys.meta_path.insert(0, finder)
        saved = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "utils" or k.startswith("utils.")}
        try:
            spec = importlib.util.spec_from_file_location(private, path)
            if spec is None or spec.loader is None:
                raise EvaluationError(f"cannot load module from {path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[private] = mod
            spec.loader.exec_module(mod)
            return mod
        finally:
            sys.meta_path.remove(finder)
            for k in [k for k in list(sys.modules) if k == "utils" or k.startswith("utils.")]:
                del sys.modules[k]
            sys.modules.update(saved)

    def score_file(self, task_id: str, prediction_path: Path) -> LawBenchScore:
        if not prediction_path.exists():
            raise EvaluationError(f"prediction file not found: {prediction_path}")
        raw = json.loads(prediction_path.read_text(encoding="utf-8"))
        try:
            data_list = [raw[str(i)] for i in range(len(raw))]
        except KeyError as exc:
            raise EvaluationError(
                f"prediction file {prediction_path} does not have "
                f"consecutive string int keys starting at '0': missing {exc}"
            ) from exc

        fn = self._get_function(task_id)
        result = fn(data_list)
        return LawBenchScore(
            task_id=task_id,
            score=float(result.get("score", 0.0)),
            abstention_rate=float(result.get("abstention_rate", 0.0)),
        )

    def score_run(self, prediction_dir: Path) -> list[LawBenchScore]:
        if not prediction_dir.is_dir():
            raise EvaluationError(f"not a directory: {prediction_dir}")
        scores: list[LawBenchScore] = []
        for p in sorted(prediction_dir.glob("*.json")):
            if p.stem in FUNCT_TABLE:
                scores.append(self.score_file(p.stem, p))
        return scores


class _DirFirstFinder(importlib.abc.MetaPathFinder):
    """Resolve selected top-level names from a specific directory, first."""

    def __init__(self, names: tuple[str, ...], directory: Path):
        self._names = set(names)
        self._directory = directory

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in self._names:
            return None
        # Handle submodules like "utils.function_utils" by delegating to
        # the standard machinery rooted at our package.
        if fullname == top:
            return importlib.machinery.PathFinder.find_spec(
                fullname, [str(self._directory.parent)], target
            )
        return importlib.machinery.PathFinder.find_spec(fullname, path, target)
