from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(_Model):
    kind: Literal["benchmark", "user"] = "benchmark"
    name: str
    version: str | None = None
    task_type: str | None = None
    sample_id: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class TaskInput(_Model):
    instruction: str
    content: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)

    @property
    def prompt(self) -> str:
        return f"{self.instruction}{self.content}"


class ParticipantSpec(_Model):
    id: str
    role: str
    kind: Literal["model", "scripted", "human"] = "model"
    profile: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


class VerificationSpec(_Model):
    verifier: str
    execution: Literal["python", "subprocess", "isolated"] = "python"
    metrics: list[str] = Field(default_factory=list)
    reference: Any | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class LexVerseTask(_Model):
    @model_validator(mode="before")
    @classmethod
    def normalize_verification(cls, value):
        from lexverse.models.verification_compat import convert_task_names
        return convert_task_names(value) if isinstance(value, dict) else value

    schema_version: Literal["1.0"] = "1.0"
    id: str
    source: SourceRef
    input: TaskInput
    participants: list[ParticipantSpec]
    verification: VerificationSpec | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    initial_conditions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("participants")
    @classmethod
    def _unique_participants(cls, participants: list[ParticipantSpec]) -> list[ParticipantSpec]:
        ids = [p.id for p in participants]
        if not ids:
            raise ValueError("a task needs at least one participant")
        if len(ids) != len(set(ids)):
            raise ValueError("participant ids must be unique")
        return participants
