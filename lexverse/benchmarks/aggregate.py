from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BenchmarkAggregator(ABC):
    @abstractmethod
    def aggregate(
        self, records: list[dict[str, Any]], *, model_name: str
    ) -> dict[str, Any]: ...

    @abstractmethod
    def write_csv(
        self, summary: dict[str, Any], path: Path, *, model_name: str
    ) -> Path: ...


def mean_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        task = str(record["task"])
        task_counts = counts.setdefault(
            task, {"expected": 0, "completed": 0, "failed": 0, "missing": 0}
        )
        task_counts["expected"] += 1
        status = str(record.get("status", "missing"))
        if status in task_counts:
            task_counts[status] += 1
        if status != "completed":
            grouped.setdefault(task, {})
            continue
        metrics = (record.get("evaluation") or {}).get("metrics") or {}
        target = grouped.setdefault(task, {})
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                target.setdefault(name, []).append(float(value))

    return {
        task: {
            "expected_samples": counts[task]["expected"],
            "completed_samples": counts[task]["completed"],
            "failed_samples": counts[task]["failed"],
            "missing_samples": counts[task]["missing"],
            "sample_count": counts[task]["completed"],
            "metrics": {
                name: sum(values) / len(values)
                for name, values in sorted(metrics.items())
                if values
            },
        }
        for task, metrics in sorted(grouped.items())
    }
