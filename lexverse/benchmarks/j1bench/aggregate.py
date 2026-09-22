from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from lexverse.benchmarks.aggregate import BenchmarkAggregator, mean_metrics


class J1BenchAggregator(BenchmarkAggregator):
    def aggregate(self, records: list[dict[str, Any]], *, model_name: str) -> dict[str, Any]:
        tasks = mean_metrics(records)
        # The upstream evaluator's scenario-level final JSON is authoritative.
        # Per-case means are only a fallback for older runs.
        for record in records:
            scenario = str(record.get("task", ""))
            aggregate = ((record.get("evaluation") or {}).get("raw_result") or {}).get(
                "aggregate"
            )
            if scenario in tasks and isinstance(aggregate, dict):
                expected = set(tasks[scenario]["metrics"])
                flattened: dict[str, float] = {}
                for key, value in aggregate.items():
                    if isinstance(value, (int, float)):
                        flattened[key] = float(value)
                    elif isinstance(value, dict):
                        flattened.update({
                            nested_key: float(nested_value)
                            for nested_key, nested_value in value.items()
                            if isinstance(nested_value, (int, float))
                        })
                tasks[scenario]["metrics"] = {
                    key: flattened[key] for key in expected if key in flattened
                }
        return {"benchmark": "j1bench", "model": model_name, "tasks": tasks}

    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["scenario", "model", "metric", "score", "case_count"])
            for scenario, result in summary["tasks"].items():
                for metric, score in result["metrics"].items():
                    writer.writerow([
                        scenario, model_name, metric, f"{score:.6f}",
                        result["sample_count"],
                    ])
        return path
