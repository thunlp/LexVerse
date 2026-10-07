from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from pydantic import BaseModel, Field, StrictInt

from .schema import LexVerseTask


class TaskSelection(BaseModel):
    task_types: list[str] | None = None
    sample_ids: list[str] | None = None
    limit_per_task: StrictInt | None = Field(default=None, ge=1)
    limits: dict[str, StrictInt | None] = Field(default_factory=dict)
    counts: dict[str, dict[str, int | None]] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, benchmark: dict, generation: dict):
        if "default_limit" in benchmark and "limit" in generation:
            raise ValueError("benchmark.default_limit conflicts with generation.limit")
        default = benchmark.get("default_limit", generation.get("limit"))
        entries = benchmark.get("tasks")
        types, limits = None, {}
        if entries is not None:
            if not isinstance(entries, list) or not entries:
                raise ValueError("benchmark.tasks must be a non-empty list")
            types = []
            for entry in entries:
                if isinstance(entry, str):
                    task_type = entry
                elif isinstance(entry, dict):
                    allowed = {"id", "limit"}
                    if benchmark.get("name") == "legalworld":
                        allowed |= {"party_role", "start_stage", "end_stage"}
                    if set(entry) - allowed:
                        raise ValueError("unknown fields in benchmark.tasks entry")
                    task_type = entry.get("id")
                else:
                    raise ValueError("task entry must be a string or mapping")
                if not isinstance(task_type, str) or not task_type:
                    raise ValueError("task id must be a non-empty string")
                if task_type in types:
                    raise ValueError(f"duplicate task type: {task_type}")
                if isinstance(entry, dict) and "limit" in entry:
                    limits[task_type] = entry["limit"]
                types.append(task_type)
        for limit in [default, *limits.values()]:
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("limit must be a positive integer or null")
        return cls(task_types=types, sample_ids=benchmark.get("cases"), limit_per_task=default, limits=limits)

    def select_types(self, available: list[str]) -> list[str]:
        if self.task_types is None:
            return list(available)
        unknown = sorted(set(self.task_types) - set(available))
        if unknown:
            raise ValueError(f"unknown task types: {unknown}")
        return list(self.task_types)

    def apply(self, task_type: str, samples: list[Any], *, sample_id: Callable) -> list[Any]:
        if self.sample_ids is not None:
            wanted = set(self.sample_ids)
            samples = [sample for sample in samples if str(sample_id(sample)) in wanted]
        available = len(samples)
        if not available:
            raise ValueError(f"no samples selected for task type {task_type}")
        limit = self.effective_limit(task_type)
        selected = samples[:limit] if limit is not None else samples
        self.counts[task_type] = {"requested": limit, "available": available, "selected": len(selected)}
        return selected

    def effective_limit(self, task_type: str) -> int | None:
        return self.limits.get(task_type, self.limit_per_task)


class TaskBundle(BaseModel):
    schema_version: int | float = 1.0
    benchmark: str
    source_version: str
    tasks: list[LexVerseTask]

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()
