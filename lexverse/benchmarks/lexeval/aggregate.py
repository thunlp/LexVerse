"""Write LexEval scores in the upstream task/model/metric/score layout."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from lexverse.benchmarks.aggregate import BenchmarkAggregator, mean_metrics
from lexverse.benchmarks.lexeval.evaluate import LexEvalScore


class LexEvalAggregator(BenchmarkAggregator):
    def aggregate(self, records: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
        tasks = mean_metrics(records)
        return {"benchmark": "lexeval", "model": model_name, "tasks": tasks}

    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            # Match upstream LexEval's public result columns.
            writer.writerow(["task", "model", "metrics", "score"])
            for task, result in summary["tasks"].items():
                for metric, score in result["metrics"].items():
                    writer.writerow([task, model_name, metric, f"{score:.6f}"])
        return path


def write_scores_csv(scores: Iterable[LexEvalScore], out_path: Path, *, model_name: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        # Upstream pandas.to_csv keeps its index column.
        writer.writerow(["", "task", "model", "metrics", "score"])
        for index, score in enumerate(scores):
            writer.writerow([
                index, score.task_id, model_name, score.metric, score.score
            ])
    return out_path
