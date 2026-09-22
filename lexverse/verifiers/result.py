from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class VerifierProvenance(BaseModel):
    name: str
    version: str
    upstream_commit: str | None = None
    config_hash: str | None = None
    model: str | None = None


class VerifierResult(BaseModel):
    status: Literal["completed", "failed", "invalid", "not_applicable"]
    metrics: dict[str, float | int] = Field(default_factory=dict)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    provenance: VerifierProvenance

    @field_validator("metrics")
    @classmethod
    def _finite_numeric_metrics(cls, metrics):
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"metric {key!r} must be a finite number")
        return metrics

