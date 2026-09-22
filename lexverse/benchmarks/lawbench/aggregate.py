"""Write LawBench scores in the upstream CSV column order."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from lexverse.benchmarks.aggregate import BenchmarkAggregator, mean_metrics
from lexverse.benchmarks.lawbench.evaluate import LawBenchScore


class LawBenchAggregator(BenchmarkAggregator):
    def aggregate(self, records: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
        tasks = mean_metrics(records)
        return {"benchmark": "lawbench", "model": model_name, "tasks": tasks}

    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            # Match upstream LawBench's task-level result columns.
            writer.writerow(["task", "model_name", "score", "abstention_rate"])
            for task, result in summary["tasks"].items():
                metrics = result["metrics"]
                writer.writerow([
                    task,
                    model_name,
                    f"{metrics.get('score', 0.0):.6f}",
                    f"{metrics.get('abstention_rate', 0.0):.6f}",
                ])
        return path


def write_scores_csv(
    scores: Iterable[LawBenchScore], out_path: Path, *, model_name: str
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["task", "model_name", "score", "abstention_rate"])
        for s in scores:
            writer.writerow(
                [s.task_id, model_name, f"{s.score:.6f}", f"{s.abstention_rate:.6f}"]
            )
    return out_path
