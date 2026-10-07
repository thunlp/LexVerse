from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class InteractionSpec(_Model):
    policy: str
    max_steps: int = Field(default=1, ge=1)
    termination: dict[str, Any] = Field(default_factory=dict)
    visibility: dict[str, Any] = Field(default_factory=dict)
    policy_config: dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(_Model):
    verifier: str
    execution: Literal["python", "subprocess", "isolated"] = "python"
    metrics: list[str] = Field(default_factory=list)
    reference: Any | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class LexVerseTask(_Model):
    schema_version: Literal[1, 1.0] = 1.0
    id: str
    source: SourceRef
    input: TaskInput
    participants: list[ParticipantSpec]
    interaction: InteractionSpec
    evaluation: EvaluationSpec
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
