from __future__ import annotations

from pydantic import BaseModel, Field


class TaskSelection(BaseModel):
    task_types: list[str] | None = None
    sample_ids: list[str] | None = None
    limit_per_task: int | None = Field(default=None, ge=1)

    def select_types(self, available: list[str]) -> list[str]:
        if self.task_types is None:
            return list(available)
        unknown = sorted(set(self.task_types) - set(available))
        if unknown:
            raise ValueError(f"unknown task types: {unknown}")
        return list(self.task_types)
