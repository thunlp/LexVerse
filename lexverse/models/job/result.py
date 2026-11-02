"""Rebuildable Job progress summary; job.json remains configuration authority."""
from typing import TypedDict


class JobResult(TypedDict):
    schema_version: str
    job_id: str
    status: str
    planned_trials: int
    statuses: dict[str, int]
    groups: dict[str, dict]
    execution_status: str
    verification_status: str
    trials: list[dict]
