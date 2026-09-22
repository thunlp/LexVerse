from __future__ import annotations

from pathlib import Path
import tempfile

from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask
from lexverse.verifiers.result import VerifierProvenance, VerifierResult

from .evaluate import J1BenchEvalConfig, J1BenchEvaluator
from .dataset import resolve_case_database


class J1BenchTaskVerifier:
    name = "j1bench"

    async def verify(self, task: LexVerseTask, result: EnvironmentResult, *, context=None):
        context = context or {}
        dialog_path = result.artifacts.get("dialog_history")
        api_key = context.get("scoring_api_key")
        output_dir = context.get("output_dir")
        if not dialog_path or not api_key or not output_dir:
            raise ValueError(
                "J1Bench verification requires result.artifacts['dialog_history'], "
                "context.scoring_api_key and context.output_dir"
            )
        scenario = task.source.task_type
        configured_paths = context.get("case_databases") or {}
        configured_path = configured_paths.get(scenario)
        if not configured_path:
            configured_path = context.get("case_database")
        case_database = (
            Path(configured_path)
            if configured_path
            else resolve_case_database(
                scenario, offline=bool(context.get("offline", False))
            )
        )
        evaluator = J1BenchEvaluator(
            scenario,
            offline=context.get("offline", False),
            python_executable=context.get("python_executable", "python"),
            case_database=case_database,
        )
        evaluated = await evaluator.evaluate(
            model_name=context.get("model_name", "LexVerse"),
            dialog_history_path=Path(dialog_path),
            output_dir=Path(output_dir),
            config=J1BenchEvalConfig(
                scoring_api_key=api_key,
                scoring_api_base=context.get("scoring_api_base"),
                scoring_model=context.get("scoring_model", "gpt-4o-mini"),
            ),
            timeout_sec=context.get("timeout_sec", 1800.0),
        )
        metrics = _metrics(evaluated.aggregate, task.evaluation.metrics)
        return VerifierResult(
            status="completed" if evaluated.per_case else "invalid",
            metrics=metrics,
            raw_result={"per_case": evaluated.per_case, "aggregate": evaluated.aggregate},
            artifacts={"final_json": str(evaluated.final_json)},
            provenance=VerifierProvenance(
                name=self.name,
                version="1",
                upstream_commit=task.source.version,
                model=context.get("scoring_model"),
            ),
        )

    async def verify_scenario(
        self,
        tasks: list[LexVerseTask],
        dialog_paths: list[Path],
        *,
        output_dir: Path,
        context: dict | None = None,
    ) -> tuple[dict[str, VerifierResult], dict]:
        """Run the upstream evaluator once over every case in one scenario."""
        if not tasks or len(tasks) != len(dialog_paths):
            raise ValueError("J1Bench scenario verification needs matching tasks and dialogs")
        context = context or {}
        scenario = tasks[0].source.task_type
        if any(task.source.task_type != scenario for task in tasks):
            raise ValueError("J1Bench scenario verification cannot mix scenarios")
        api_key = context.get("scoring_api_key")
        if not api_key:
            raise ValueError("J1Bench verification requires context.scoring_api_key")

        configured_paths = context.get("case_databases") or {}
        configured_path = configured_paths.get(scenario) or context.get("case_database")
        case_database = (
            Path(configured_path)
            if configured_path
            else resolve_case_database(scenario, offline=bool(context.get("offline", False)))
        )
        evaluator = J1BenchEvaluator(
            scenario,
            offline=context.get("offline", False),
            python_executable=context.get("python_executable", "python"),
            case_database=case_database,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False,
            dir=output_dir,
        ) as combined:
            combined_path = Path(combined.name)
            for path in dialog_paths:
                text = path.read_text(encoding="utf-8")
                combined.write(text)
                if text and not text.endswith("\n"):
                    combined.write("\n")
        try:
            evaluated = await evaluator.evaluate(
                model_name=context.get("model_name", "LexVerse"),
                dialog_history_path=combined_path,
                output_dir=output_dir,
                config=J1BenchEvalConfig(
                    scoring_api_key=api_key,
                    scoring_api_base=context.get("scoring_api_base"),
                    scoring_model=context.get("scoring_model", "gpt-4o-mini"),
                ),
                timeout_sec=context.get("timeout_sec", 1800.0),
            )
        finally:
            combined_path.unlink(missing_ok=True)

        by_case = {
            str(item.get("case_id")): item
            for item in evaluated.per_case
            if isinstance(item, dict) and item.get("case_id")
        }
        results: dict[str, VerifierResult] = {}
        for task in tasks:
            case_id = task.source.sample_id
            raw_case = by_case.get(case_id)
            results[case_id] = VerifierResult(
                status="completed" if raw_case is not None else "invalid",
                metrics=_metrics(raw_case or {}, task.evaluation.metrics),
                raw_result={"per_case": raw_case or {}, "aggregate": evaluated.aggregate},
                artifacts={"final_json": str(evaluated.final_json)},
                provenance=VerifierProvenance(
                    name=self.name,
                    version="1",
                    upstream_commit=task.source.version,
                    model=context.get("scoring_model"),
                ),
            )
        return results, evaluated.aggregate


def _metrics(aggregate: dict, expected: list[str]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in aggregate.items():
        if isinstance(value, (int, float)):
            flattened[key] = float(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (int, float)):
                    flattened[nested_key] = float(nested_value)
    return {name: flattened[name] for name in expected if name in flattened}
